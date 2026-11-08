"""PUCT search: open-loop MCTS with selective depth over the tiles value (``planner_pv2_puct``).

Predeclared in ``experiments/puct-search.json`` (2026-11-08) before any code ran a game,
with ``amendment_1_2026_11_08`` (before any screen game): priors are a softmax over the
served scores standardized within the decision; inside simulations every seat plays the
served bot with player-trade offers disabled (``OfferFreeServedPlanner``; answers to
offers and everything else as served); the search is root-parallel (``SHARDS`` = 4
independent trees, shard ``k`` owning simulation indices ``k, k + 4, ...``, run in 4
worker processes and merged at the root by summing visit counts and value sums, so with
``max_simulations`` set the result does not depend on the number of workers); a decision
that falls back hands the rest of the turn to ``planner_pv2_tiles`` with no new
searches, and at most 4 decisions per turn are searched.

The tree holds only our own decisions (open loop): a node is reached by the sequence
of our choices, whatever the dice and the opponents did in between. Each simulation

1. samples the hidden information at the root decision from our view only: every
   opponent's hand is a multinomial draw from its card-counted estimate with its exact
   public card count (and the bank's supply respected), and the unseen development
   cards (the deck and the opponents' hands) are a shuffle of the remaining-card prior
   (the 25-card deck minus our own cards and every card played in public); the dice
   and steals follow a per-simulation RNG seeded from a hash of the decision and the
   simulation index;
2. descends: at each of our decision points the candidates available in this sampled
   pass are scored by PUCT, ``Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))``
   with ``c_puct`` 1.5, priors a softmax over the served bot's own scores
   (temperature 1), and unvisited children valued at the parent's mean (first-play
   urgency = parent value); candidates not legal in this pass are skipped;
3. at a child never visited before, evaluates the end-of-our-turn afterstate with the
   ``planner_pv2_tiles`` value net (P(I win) on the encoder-v2 afterstate from our
   view and card-counted belief) and backs up; otherwise plays the choice, ends the
   turn, lets every other decision be made by the served ``planner_offer_accept``
   (our own robber, discards and trade answers included) until our next decision or
   the end of the game (terminal: 1 if we won, else 0), and descends again.

Our decisions are whole-turn plans (``end now`` plus the served bot's top-4 sequences,
as ``planner_pv2_tiles``) and, in the setup, settlements (the served bot's top-8
vertices by its placement value; the road follows the served bot). The root move is
the most-visited child (ties: higher mean value, then the served order). Budgets are
wall-clock (1.2 s per turn plan, 1.5 s per setup settlement); fewer than 16 completed
simulations fall back to ``planner_pv2_tiles``' own choice and are counted. A chosen
turn plan is then played step by step while each step stays legal; when it is
exhausted the served end-of-turn path runs (as ``planner_pv2_tiles`` does when it
ends a turn).
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
import os
import random
import time

from .card_planner import _fold, resource_estimates
from .encoding import encode
from .engine import Resource, longest_road
from .observation import Observation
from .pv_search import DECK_PRIOR, REPLACEABLE, PVSearchTilesPlanner, PVSearchV2BFixPlanner, served_sequences
from .tactical import public_projection
from .unified_planner import AcceptanceOfferPlanner

C_PUCT = 1.5
TURN_K = 4
SETUP_K = 8
TURN_BUDGET = 1.2
SETUP_BUDGET = 1.5
MIN_SIMULATIONS = 16
SIMULATION_CAP = 4096       # safety bound on simulations per decision (not expected to bind)
OVERRUN_SECONDS = .25       # a simulation still running this long after the budget is abandoned
ADVANCE_STEP_CAP = 1500     # engine steps between two of our decisions before a pass is cut (counted)
SHARDS = 4                  # independent root-parallel trees (simulation index k + 4j -> shard k)
WORKERS = 4                 # worker processes running the shards (1: in-process, same result)
MAX_SEARCHES_PER_TURN = 4
_RESOURCES = list(Resource)


class Abandoned(Exception):
    """A simulation ran past its deadline; it is discarded without backup."""


# ----------------------------------------------------------------- tree core
class Node:
    """Edge statistics of the choice leading here (``N``, ``W``) and the number of
    selections made at this node (``visits``, the PUCT ``N_parent``)."""

    __slots__ = ('N', 'W', 'visits', 'children')

    def __init__(self):
        self.N, self.W, self.visits, self.children = 0, 0., 0, {}

    @property
    def mean(self):
        return self.W / self.N if self.N else None


def softmax(scores):
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def standardized_priors(scores):
    """Softmax (temperature 1) of the scores standardized within the decision (z-scores,
    population SD); equal scores give a uniform prior."""
    n = len(scores)
    mean = sum(scores) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in scores) / n)
    if not sd > 0:
        return [1 / n] * n
    return softmax([(x - mean) / sd for x in scores])


def entropy(priors):
    return -sum(p * math.log(p) for p in priors if p > 0)


class OfferFreeServedPlanner(AcceptanceOfferPlanner):
    """The served bot with player-trade offers disabled: the opponent model (and our own
    non-plan decisions) inside simulations. It still answers offers; all else is served."""

    offers = False


def puct_select(node, candidates, c_puct=C_PUCT, fpu=None):
    """Index of the PUCT choice among ``candidates`` = [(key, prior), ...] (served order).

    ``Q + c_puct * P * sqrt(N_parent) / (1 + N)``; an unvisited child's ``Q`` is ``fpu``
    (the parent's mean). Ties: higher prior, then earlier in served order.
    """
    if fpu is None:
        fpu = node.mean if node.mean is not None else .5
    root = math.sqrt(node.visits)
    best, best_key = None, None
    for i, (key, prior) in enumerate(candidates):
        child = node.children.get(key)
        n = child.N if child is not None else 0
        q = child.W / n if n else fpu
        rank = (q + c_puct * prior * root / (1 + n), prior, -i)
        if best_key is None or rank > best_key:
            best, best_key = i, rank
    return best


def principal_depth(root):
    """Edges on the most-visited line from the root (children with at least one visit)."""
    depth, node = 0, root
    while True:
        visited = [c for c in node.children.values() if c.N > 0]
        if not visited:
            return depth
        node = max(visited, key=lambda c: c.N)
        depth += 1


def simulate(env, root, index, deadline=None, clock=time.perf_counter):
    """One simulation; returns its depth (our decisions on its path).

    ``env`` provides ``sample(index) -> state``, ``candidates(state) -> [(key, prior,
    payload)]`` (available choices, served order), ``play(state, payload, expand) ->
    (state, value)`` (``value`` is not None at a terminal or, when ``expand``, the
    evaluation of the afterstate) and ``advance(state, deadline) -> (state, value)``
    (to our next decision; ``value`` is not None at a terminal or a cut pass).
    """
    state = env.sample(index)
    node, path, selected = root, [root], []
    while True:
        if deadline is not None and clock() > deadline:
            raise Abandoned
        candidates = env.candidates(state)
        if not candidates:
            value = env.evaluate(state)
            break
        pick = puct_select(node, [(k, p) for k, p, _ in candidates])
        key, _, payload = candidates[pick]
        selected.append(node)
        child = node.children.get(key)
        expand = child is None or child.N == 0
        state, value = env.play(state, payload, expand)
        if child is None:
            child = node.children[key] = Node()
        path.append(child)
        if value is not None:
            break
        state, value = env.advance(state, deadline)
        if value is not None:
            break
        node = child
    for n in selected:
        n.visits += 1
    for n in path:
        n.N += 1
        n.W += value
    return len(path) - 1


def run_search(env, budget=None, max_simulations=None, clock=time.perf_counter, offset=0, stride=1):
    """Simulate until ``max_simulations`` (deterministic) or the wall-clock ``budget``.

    Simulation ``j`` of this tree uses index ``offset + stride * j``; with
    ``max_simulations`` the tree runs every such index below ``max_simulations``.
    """
    root = Node()
    started = clock()
    deadline = None if max_simulations is not None else started + budget + OVERRUN_SECONDS
    done, max_depth, abandoned = 0, 0, 0
    while True:
        index = offset + stride * done
        if max_simulations is not None:
            if index >= max_simulations:
                break
        elif clock() - started >= budget or done >= SIMULATION_CAP:
            break
        try:
            depth = simulate(env, root, index, deadline, clock)
        except Abandoned:
            abandoned += 1
            break
        done += 1
        max_depth = max(max_depth, depth)
    return root, {'simulations': done, 'max_depth': max_depth, 'pv_depth': principal_depth(root),
                  'abandoned': abandoned}


def best_child(root, order):
    """Most-visited root child; ties: higher mean value, then earlier in ``order``."""
    ranked = [(root.children[k].N, root.children[k].mean or 0., -i, k)
              for i, k in enumerate(order) if k in root.children]
    return max(ranked)[3] if ranked else None


# ------------------------------------------------------- worlds from the view
def unseen_cards(view, seat):
    """The development cards we have not seen: the 25-card deck minus our own and every
    card played in public (the deck plus the opponents' hidden cards)."""
    counts = dict(DECK_PRIOR)
    for card in view.players[seat].development or ():
        counts[card] -= 1
    for event in view.events:
        action = event['action']
        if action['type'] == 'play_development':
            counts[action['card']] -= 1
    return [card for card, n in DECK_PRIOR for _ in range(max(0, counts[card]))]


def base_world(view):
    """The engine state at our decision with every opponent's hand and development cards
    empty and no deck: public information plus our own hand only."""
    game = public_projection(view)
    game.robber_tile = view.robber_tile
    game.rolled = view.rolled
    game.setup_index = view.setup_index
    game.pending_setup_vertex = view.pending_setup_vertex
    game.discard_remaining = dict(view.discard_remaining)
    game.robber_actor = view.robber_actor
    game.trade_offer = deepcopy(view.trade_offer)
    game.current = view.seat
    occupied = game.occupied()
    game.road_lengths = [longest_road(set(p.roads), {v for v, o in occupied.items() if o != s})
                         for s, p in enumerate(game.players)]
    game.deck = []
    game.events = []
    return game


def sample_hand(rng, estimate, total, supply):
    """``total`` cards drawn multinomially from ``estimate`` without exceeding ``supply``."""
    hand = {r: 0 for r in _RESOURCES}
    for _ in range(total):
        weights = [max(estimate[r], 0.) if supply[r] > 0 else 0. for r in _RESOURCES]
        if sum(weights) <= 0:
            weights = [1. if supply[r] > 0 else 0. for r in _RESOURCES]
            if sum(weights) <= 0:
                break
        r = rng.choices(_RESOURCES, weights)[0]
        hand[r] += 1
        supply[r] -= 1
    return hand


def light_copy(world):
    """A copy without the event history and without recording (legality checks, leaves)."""
    copy = deepcopy(world, {id(world.board): world.board, id(world.events): []})
    copy.record_events = False
    return copy


def replay(world, seat, seq):
    """Apply ``seq`` in place; False if a step is not legal (a purchase ends the plan)."""
    for action in seq:
        if action not in world.legal_actions(seat):
            return False
        world.apply(seat, action)
        if action['type'] == 'buy_development' or world.winner is not None:
            break
    return True


def env_inputs(view, seat, salt, root_candidates):
    """Everything a search needs, derived from ``seat``'s view only (sent to the workers)."""
    key = (salt, view.turn, view.phase, view.setup_index, seat, len(view.events),
           sorted((r.value, n) for r, n in view.players[seat].resources.items()),
           list(view.players[seat].development or ()))
    return {'seat': seat, 'base': base_world(view), 'estimates': resource_estimates(view.events),
            'unseen': unseen_cards(view, seat), 'counts': [p.resource_count for p in view.players],
            'dev_counts': [p.development_count for p in view.players], 'root_candidates': root_candidates,
            'key': hashlib.sha256(json.dumps(key).encode()).hexdigest()}


class CatanEnv:
    """The search environment of one decision, built from ``env_inputs`` (view-derived)."""

    def __init__(self, planner, inputs):
        self.planner, self.seat = planner, inputs['seat']
        self.bots = planner.served_bots()
        self.base, self.estimates, self.unseen = inputs['base'], inputs['estimates'], inputs['unseen']
        self.counts, self.dev_counts = inputs['counts'], inputs['dev_counts']
        self.root_candidates, self.key = inputs['root_candidates'], inputs['key']
        self.truncated = 0
        self.prior_entropy_sum, self.prior_count = 0., 0

    # -- sampling
    def sample(self, index):
        rng = random.Random(int(hashlib.sha256(f'{self.key}|{index}'.encode()).hexdigest()[:16], 16))
        world = deepcopy(self.base, {id(self.base.board): self.base.board})
        supply = {r: 19 - world.players[self.seat].resources[r] for r in _RESOURCES}
        for s, player in enumerate(world.players):
            if s != self.seat:
                player.resources = sample_hand(rng, self.estimates[s], self.counts[s], supply)
        cards = list(self.unseen)
        rng.shuffle(cards)
        for s, player in enumerate(world.players):
            if s != self.seat:
                held = [cards.pop() if cards else 'knight' for _ in range(self.dev_counts[s])]
                player.development = [(card, -1) for card in held]
        world.deck = cards
        world.rng = random.Random(rng.getrandbits(64))
        world.record_events = True
        for bot in self.bots:
            bot.rng = random.Random(rng.getrandbits(64))
            bot.last_offer_turn = -1
        return {'world': world, 'belief': self.estimates, 'root': True}

    # -- decisions
    def candidates(self, state):
        world = state['world']
        if world.winner is not None:
            return []
        if state.get('root'):
            ranked = self.root_candidates
        else:
            view = Observation.from_game(world, self.seat, include_events=False)
            ranked = (self.planner.setup_candidates(view, self.seat) if world.phase == 'setup_settlement'
                      else self.planner.turn_candidates(view, self.seat))
        available = []
        for key, score, plan in ranked:
            after = light_copy(world)
            if self._play_plan(after, plan):
                available.append((key, score, (plan, after)))
        if not available:
            return []
        priors = standardized_priors([score for _, score, _ in available])
        self.prior_entropy_sum += entropy(priors)
        self.prior_count += 1
        return [(key, p, payload) for (key, _, payload), p in zip(available, priors)]

    def _play_plan(self, world, plan):
        if isinstance(plan, int):  # setup settlement, then the served road
            action = {'type': 'build_settlement', 'vertex': plan}
            if action not in world.legal_actions(self.seat):
                return False
            world.apply(self.seat, action)
            bot = self.bots[self.seat]
            world.apply(self.seat, bot.choose(Observation.for_policy(world, self.seat, bot), self.seat))
            return True
        return replay(world, self.seat, plan)

    def play(self, state, payload, expand):
        plan, after = payload
        if after.winner is not None:
            return state, float(after.winner == self.seat)
        if expand:
            return state, self.planner.leaf_value(after, self.seat, self.belief(state))
        world = state['world']
        if not self._play_plan(world, plan):
            raise RuntimeError('plan legal on the copy but not on the world')
        return {**state, 'root': False}, None

    def advance(self, state, deadline=None):
        world, seat, steps = state['world'], self.seat, 0
        # after a turn plan our turn is still open (actions, or free roads to place): end it
        open_turn = world.current == seat and world.phase in ('actions', 'free_roads')
        while world.winner is None:
            s = world.current
            if s == seat and world.phase == 'actions' and open_turn:
                action, open_turn = {'type': 'end_turn'}, False
            elif s == seat and world.phase in ('actions', 'setup_settlement'):
                return state, None
            else:
                bot = self.bots[s]
                action = bot.choose(Observation.for_policy(world, s, bot), s)
            world.apply(s, action)
            steps += 1
            if steps > ADVANCE_STEP_CAP:
                self.truncated += 1
                return state, self.evaluate(state)
            if deadline is not None and steps % 16 == 0 and time.perf_counter() > deadline:
                raise Abandoned
        return state, float(world.winner == seat)

    # -- leaf value
    def belief(self, state):
        """Card-counted opponent hands: the root estimate folded with this pass's events."""
        world, hands = state['world'], [dict(h) for h in state['belief']]
        for event in world.events:
            _fold(hands, event['public'] | event['private'].get(self.seat, {}))
        return hands

    def evaluate(self, state):
        world = state['world']
        if world.winner is not None:
            return float(world.winner == self.seat)
        return self.planner.leaf_value(world, self.seat, self.belief(state))


# ------------------------------------------------------------------ the bot
# ------------------------------------------------------- root parallelism
def run_shard(planner, inputs, shard, budget, max_simulations):
    """Shard ``shard``'s tree; returns its root statistics (picklable)."""
    env = CatanEnv(planner, inputs)
    root, stats = run_search(env, budget=budget, max_simulations=max_simulations, offset=shard, stride=SHARDS)
    children = {k: (c.N, c.W, principal_depth(c)) for k, c in root.children.items()}
    return {'children': children, 'N': root.N, 'W': root.W, 'stats': stats, 'truncated': env.truncated,
            'prior_entropy_sum': env.prior_entropy_sum, 'prior_count': env.prior_count}


_WORKER_PLANNER = None
_POOL = None


def _worker_init():
    global _WORKER_PLANNER
    _WORKER_PLANNER = PUCTSearchPlanner(random.Random(0), pool=False)


def _worker_shard(args):
    return run_shard(_WORKER_PLANNER, *args)


def _worker_ready(delay):
    time.sleep(delay)
    return os.getpid()


def search_pool(workers):
    """A process-wide pool of search workers (spawned once, warmed before first use)."""
    global _POOL
    identity = (workers, os.environ.get('CATAN_PARAMETER_MODEL'), os.environ.get('CATAN_ENGINE'))
    if _POOL is not None and _POOL[0] != identity:
        _POOL[1].shutdown(wait=False, cancel_futures=True)
        _POOL = None
    if _POOL is None:
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_worker_init)
        list(pool.map(_worker_ready, [.2] * workers))  # every worker has loaded its planner
        _POOL = (identity, pool)
    return _POOL[1]


def merge_shards(results):
    """Sum the shards' root statistics in shard order (independent of the worker count)."""
    root = Node()
    depths = {}
    for r in results:
        root.N += r['N']
        root.W += r['W']
        for key, (n, w, depth) in r['children'].items():
            child = root.children.setdefault(key, Node())
            child.N += n
            child.W += w
            if n:
                depths.setdefault(key, []).append(depth)
    return root, depths


def _key(seq):
    return json.dumps(seq, sort_keys=True)


class PUCTSearchPlanner(PVSearchTilesPlanner):
    """``planner_pv2_puct``: see the module docstring and ``experiments/puct-search.json``."""

    turn_budget = TURN_BUDGET
    setup_budget = SETUP_BUDGET
    max_simulations = None  # tests: a fixed simulation count instead of the clock
    workers = WORKERS

    def __init__(self, rng=None, pool=True):
        super().__init__(rng)
        # Seed of every search stream, read from the planner's RNG state without consuming it.
        state = self.rng.getstate() if self.rng is not None else None
        self.search_salt = hashlib.sha256(repr(state).encode()).hexdigest()[:16]
        self.diagnostics = []
        self._bots = None
        self._plan = None
        self._ended = None
        self._searches = None      # (turn, seat, searched decisions this turn)
        self._tiles_turn = None    # (turn, seat): planner_pv2_tiles plays the rest of this turn
        self._use_pool = pool
        if pool and self.workers > 1:
            search_pool(self.workers)  # spawn and warm the workers before the first timed decision

    def served_bots(self):
        if self._bots is None:
            self._bots = [OfferFreeServedPlanner(random.Random(s + 1)) for s in range(4)]
        return self._bots

    @property
    def fallbacks(self):
        return sum(d['fallback'] for d in self.diagnostics)

    # -- candidates (served scores are the priors' logits)
    def turn_candidates(self, view, seat):
        root = public_projection(view)
        end = ('[]', self._leaf(root, seat), [])
        rows = [end] + [(_key(seq), score, seq) for score, seq in served_sequences(self, view, seat)[:TURN_K]]
        return sorted(rows, key=lambda row: -row[1])  # stable: the served bot's preference order

    def setup_candidates(self, view, seat):
        root = public_projection(view)

        def placement(action):
            child = deepcopy(root, {id(root.board): root.board})
            child.players[seat].settlements.add(action['vertex'])
            return self._value(child, seat)
        legal = view.legal_actions(seat)
        scored = sorted(((placement(a), i, a['vertex']) for i, a in enumerate(legal)), key=lambda t: (-t[0], t[1]))
        return [(f'v{v}', score, v) for score, _, v in scored[:SETUP_K]]

    # -- leaf value: the tiles net on the afterstate from our view and belief
    def leaf_value(self, world, seat, hands):
        view = Observation.from_game(world, seat, include_events=False)
        belief = public_projection(view)
        belief.robber_tile = world.robber_tile
        for s, player in enumerate(belief.players):
            if s != seat:
                player.resources = {r: hands[s][r] for r in _RESOURCES}
                player.development = [('knight', -1)] * view.players[s].development_count
        logit = float(self.net.logits([encode(view, belief)])[0])
        return 1 / (1 + math.exp(-logit))

    # -- search
    def search(self, view, seat, candidates, budget):
        started = time.perf_counter()
        inputs = env_inputs(view, seat, self.search_salt, candidates)
        jobs = [(inputs, k, budget, self.max_simulations) for k in range(SHARDS)]
        if self._use_pool and self.workers > 1:
            results = list(search_pool(self.workers).map(_worker_shard, jobs))
        else:
            # in-process: shards run one after another, each on a quarter of the clock
            results = [run_shard(self, inputs, k, budget / SHARDS, self.max_simulations) for k in range(SHARDS)]
        root, depths = merge_shards(results)
        order = [key for key, _, _ in candidates]
        pick = best_child(root, order)
        simulations = sum(r['stats']['simulations'] for r in results)
        fallback = simulations < MIN_SIMULATIONS or pick is None
        priors = standardized_priors([score for _, score, _ in candidates])
        prior_count = sum(r['prior_count'] for r in results)
        pv = depths.get(pick, [0])
        self.diagnostics.append({
            'reason': 'fallback' if fallback else 'puct', 'phase': view.phase, 'turn': view.turn,
            'seconds': time.perf_counter() - started, 'rollouts': simulations,
            'truncated': sum(r['truncated'] for r in results), 'simulations': simulations,
            'max_depth': max(r['stats']['max_depth'] for r in results),
            'pv_depth': 1 + max(pv), 'pv_depth_mean': 1 + sum(pv) / len(pv),
            'abandoned': sum(r['stats']['abandoned'] for r in results), 'fallback': fallback,
            'candidates': len(candidates), 'prior_entropy': entropy(priors), 'prior_top_mass': max(priors),
            'mean_prior_entropy': (sum(r['prior_entropy_sum'] for r in results) / prior_count
                                   if prior_count else None),
            'shard_simulations': [r['stats']['simulations'] for r in results],
            'visits': [root.children[k].N if k in root.children else 0 for k in order],
            'choice': order.index(pick) if pick is not None else None})
        return None if fallback else next(plan for key, _, plan in candidates if key == pick)

    def _end_path(self, view, seat):
        legal = view.legal_actions(seat)
        action = (self._fallback(view, seat) if any(a['type'] == 'play_development' for a in legal)
                  else {'type': 'end_turn'})
        return self._end_of_turn_offer(view, seat, action)

    def _hand(self, view, seat):
        return (view.turn, seat, tuple(sorted((r.value, n) for r, n in view.players[seat].resources.items())))

    def choose(self, view, seat):
        if view.phase == 'setup_settlement' and len(view.actions) > 1:
            candidates = self.setup_candidates(view, seat)
            plan = self.search(view, seat, candidates, self.setup_budget) if len(candidates) > 1 else None
            if plan is None:
                return AcceptanceOfferPlanner.choose(self, view, seat)  # planner_pv2_tiles' setup choice
            return {'type': 'build_settlement', 'vertex': plan}
        if view.phase == 'actions' and self._plan is not None and self._plan[0] == (view.turn, seat):
            remaining = self._plan[1]
            if remaining and remaining[0] in view.legal_actions(seat):
                self._plan = ((view.turn, seat), remaining[1:])
                return remaining[0]
            self._plan = None
            if not remaining:
                self._ended = self._hand(view, seat)
                return self._end_path(view, seat)
        elif self._plan is not None and self._plan[0] != (view.turn, seat):
            self._plan = None
        if view.phase == 'actions' and self._tiles_turn == (view.turn, seat):
            return PVSearchV2BFixPlanner.choose(self, view, seat)  # rest of a fallen-back turn
        before = self.last_offer_turn
        served = AcceptanceOfferPlanner.choose(self, view, seat)
        if (view.phase != 'actions' or served['type'] in ('offer_trade', 'play_development')
                or served['type'] not in REPLACEABLE or not any(a['type'] == 'end_turn' for a in view.actions)):
            return served
        self.last_offer_turn = before
        if self._ended == self._hand(view, seat):
            return served if served['type'] == 'end_turn' else self._end_path(view, seat)
        candidates = self.turn_candidates(view, seat)
        plan = []
        if len(candidates) > 1:
            if self._searches is None or self._searches[:2] != (view.turn, seat):
                self._searches = (view.turn, seat, 0)
            if self._searches[2] >= MAX_SEARCHES_PER_TURN:
                self.diagnostics.append({'reason': 'turn_cap', 'phase': view.phase, 'turn': view.turn,
                                         'seconds': 0., 'rollouts': 0, 'truncated': 0, 'fallback': False})
                plan = None
            else:
                self._searches = (view.turn, seat, self._searches[2] + 1)
                plan = self.search(view, seat, candidates, self.turn_budget)
        if plan is None:  # fallback or turn cap: planner_pv2_tiles plays the rest of this turn
            self._tiles_turn = (view.turn, seat)
            self.last_offer_turn = before
            return PVSearchV2BFixPlanner.choose(self, view, seat)
        if not plan:
            self._ended = self._hand(view, seat)
            return served if served['type'] == 'end_turn' else self._end_path(view, seat)
        self._plan = ((view.turn, seat), plan[1:])
        return plan[0]
