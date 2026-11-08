"""``planner_pv2_puct``: PUCT arithmetic, open-loop availability, selective depth,
determinism, the hidden-information boundary, legality and fallback counting."""
from copy import deepcopy
import json
import math

import pytest

np = pytest.importorskip('numpy')

from catan_ai import puct_search as ps  # noqa: E402
from catan_ai import tiles_value as tv  # noqa: E402
from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game, Resource  # noqa: E402
from catan_ai.experiments import source_files  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from catan_ai.parameter_planner import DEFAULT  # noqa: E402

needs_weights = pytest.mark.skipif(not tv.TILES_V1.with_suffix('.npz').exists(),
                                   reason='tiles weights are a local artifact (not committed)')


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


# ---------------------------------------------------------------- tree core
def test_puct_arithmetic_on_a_hand_example():
    node = ps.Node()
    node.visits, node.N, node.W = 9, 10, 5.  # parent mean .5
    a, b = ps.Node(), ps.Node()
    a.N, a.W = 6, 4.2   # Q .7
    b.N, b.W = 3, .6    # Q .2
    node.children = {'a': a, 'b': b}
    priors = [('a', .5), ('b', .3), ('c', .2)]
    # U = c * P * sqrt(9) / (1 + N): a .7 + 1.5*.5*3/7, b .2 + 1.5*.3*3/4, c (FPU .5) + 1.5*.2*3/1
    scores = [.7 + 1.5 * .5 * 3 / 7, .2 + 1.5 * .3 * 3 / 4, .5 + 1.5 * .2 * 3]
    assert scores[2] > scores[0] > scores[1]
    assert ps.puct_select(node, priors) == 2
    # with c unavailable in this pass, a wins
    assert ps.puct_select(node, priors[:2]) == 0
    # first visit of a node: sqrt(0) = 0, all unvisited -> highest prior, then served order
    fresh = ps.Node()
    assert ps.puct_select(fresh, [('x', .2), ('y', .5), ('z', .5)]) == 1
    assert ps.softmax([0., math.log(3.)]) == pytest.approx([.25, .75])


class ChainEnv:
    """Synthetic open-loop environment: each of our decisions picks 'good' or 'bad'.

    An afterstate is worth the mean of its picks (good .8, bad .2); the game ends after ``horizon``
    of our decisions with a win iff every pick was good. ``flaky`` is unavailable in
    odd-numbered passes (availability-aware selection must skip it).
    """

    def __init__(self, horizon=8, flaky=False):
        self.horizon, self.flaky = horizon, flaky
        self.offered = []

    def sample(self, index):
        return {'path': (), 'index': index}

    def candidates(self, state):
        keys = [('good', .4), ('bad', .4)]
        if self.flaky:
            keys.append(('flaky', .2))
            if state['index'] % 2:
                keys = keys[:2]
        total = sum(p for _, p in keys)
        self.offered.append((state['index'], len(state['path']), [k for k, _ in keys]))
        return [(k, p / total, k) for k, p in keys]

    def play(self, state, payload, expand):
        path = state['path'] + (payload,)
        after = {**state, 'path': path}
        if len(path) >= self.horizon:
            return after, float(all(p == 'good' for p in path))
        if expand:  # the mean worth of every pick so far: one bad pick caps the line
            worth = {'good': .8, 'bad': .2, 'flaky': .5}
            return after, sum(worth[p] for p in path) / len(path)
        return after, None

    def advance(self, state, deadline=None):
        return state, None

    def evaluate(self, state):
        return .5


def _depth_under(node):
    visited = [c for c in node.children.values() if c.N]
    return 1 + max(map(_depth_under, visited)) if visited else 0


def test_selective_depth_the_better_branch_gets_more_visits_and_a_deeper_subtree():
    root, stats = ps.run_search(ChainEnv(horizon=10), max_simulations=200)
    good, bad = root.children['good'], root.children['bad']
    assert good.N > 2 * bad.N
    assert _depth_under(good) > _depth_under(bad)
    assert stats['simulations'] == 200 and stats['max_depth'] >= 6 and stats['pv_depth'] >= 6
    assert ps.best_child(root, ['good', 'bad']) == 'good'
    assert root.N == 200 and root.visits == 200


def test_open_loop_skips_children_unavailable_in_the_current_pass():
    env = ChainEnv(horizon=4, flaky=True)
    root, _ = ps.run_search(env, max_simulations=120)
    flaky = root.children['flaky']
    assert flaky.N > 0
    # the root offered 'flaky' only in even passes, so it was picked only there
    even_root_passes = sum(1 for i, depth, keys in env.offered if depth == 0 and 'flaky' in keys)
    assert flaky.N <= even_root_passes == 60
    assert sum(c.N for c in root.children.values()) == root.N == 120


def test_search_is_deterministic_for_a_fixed_simulation_count():
    def run():
        root, stats = ps.run_search(ChainEnv(horizon=6, flaky=True), max_simulations=64)
        return stats, sorted((k, c.N, round(c.W, 9)) for k, c in root.children.items())
    assert run() == run()


# ---------------------------------------------------------- catan worlds
def _game_at_our_turn_plan(seed, turn=30, seat=0, policy='planner_offer_accept'):
    """A seeded game advanced to ``seat``'s first actions decision at or after ``turn``."""
    game = Game.new(list('abcd'), seed)
    bots = [make_policy(policy, seed * 10 + s) for s in range(4)]
    while game.winner is None:
        s = game.current
        if (s == seat and game.phase == 'actions' and game.turn >= turn
                and len(game.legal_actions(s)) > 3):
            return game
        game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    raise AssertionError('game ended before the decision')


def test_sampled_worlds_keep_public_counts_and_our_hand():
    game = _game_at_our_turn_plan(2950007)
    view = Observation.from_game(game, 0)
    env_planner = type('P', (), {'served_bots': lambda self: [None] * 4, 'search_salt': 'x'})()
    env = ps.CatanEnv(env_planner, view, 0, [])
    buys = sum(e['action']['type'] == 'buy_development' for e in view.events)
    for index in range(20):
        env.bots = [type('B', (), {})() for _ in range(4)]
        world = env.sample(index)['world']
        assert world.players[0].resources == game.players[0].resources
        assert world.players[0].development == game.players[0].development
        for s in range(1, 4):
            assert sum(world.players[s].resources.values()) == sum(game.players[s].resources.values())
            assert len(world.players[s].development) == len(game.players[s].development)
        assert all(n >= 0 for n in world.bank.values())
        assert len(world.deck) == len(game.deck) == 25 - buys
        public = lambda g: [a for a in g.legal_actions(0) if a['type'] != 'bank_trade']  # noqa: E731
        assert public(world) == public(game)
        assert world.phase == game.phase and world.current == game.current and world.turn == game.turn


@needs_weights
def test_hidden_information_cannot_change_the_decision(monkeypatch):
    monkeypatch.setattr(ps.PUCTSearchPlanner, 'max_simulations', 24)
    game = _game_at_our_turn_plan(2950011)
    twin = deepcopy(game)
    # same public state, different hidden hands and development deck order
    for s in (1, 2, 3):
        hand = twin.players[s].resources
        total = sum(hand.values())
        twin.players[s].resources = {r: 0 for r in Resource}
        twin.players[s].resources[list(Resource)[s]] = total
    twin.deck.reverse()
    twin.rng.seed(12345)
    assert Observation.from_game(game, 0) == Observation.from_game(twin, 0)
    choices = []
    for g in (game, twin):
        planner = make_policy('planner_pv2_puct', 77)
        action = planner.choose(Observation.for_policy(g, 0, planner), 0)
        choices.append((action, [(d['visits'], d['simulations'], d['max_depth']) for d in planner.diagnostics]))
    assert choices[0] == choices[1]
    assert choices[0][1], 'the decision must have been searched'


@needs_weights
def test_deterministic_decision_with_max_simulations(monkeypatch):
    monkeypatch.setattr(ps.PUCTSearchPlanner, 'max_simulations', 20)
    game = _game_at_our_turn_plan(2950013)
    out = []
    for _ in range(2):
        planner = make_policy('planner_pv2_puct', 5)
        action = planner.choose(Observation.for_policy(game, 0, planner), 0)
        out.append((action, [{k: v for k, v in d.items() if k != 'seconds'} for d in planner.diagnostics]))
    assert out[0] == out[1]


@needs_weights
def test_fallback_to_pv2_tiles_is_counted_when_too_few_simulations(monkeypatch):
    monkeypatch.setattr(ps.PUCTSearchPlanner, 'max_simulations', ps.MIN_SIMULATIONS - 1)
    game = _game_at_our_turn_plan(2950017)
    planner = make_policy('planner_pv2_puct', 9)
    action = planner.choose(Observation.for_policy(game, 0, planner), 0)
    control = make_policy('planner_pv2_tiles', 9)
    assert action == control.choose(Observation.for_policy(game, 0, control), 0)
    assert planner.fallbacks == 1
    record = planner.diagnostics[-1]
    assert record['fallback'] and record['reason'] == 'fallback'
    assert record['simulations'] == ps.MIN_SIMULATIONS - 1


@needs_weights
def test_full_seeded_game_against_tactical_plays_only_legal_moves(monkeypatch):
    monkeypatch.setattr(ps.PUCTSearchPlanner, 'max_simulations', ps.MIN_SIMULATIONS)
    game = Game.new(list('abcd'), 2950021)
    bots = [make_policy('planner_pv2_puct', 1)] + [make_policy('tactical', i) for i in (2, 3, 4)]
    while game.winner is None and game.turn < 1000:
        s = game.current
        view = Observation.for_policy(game, s, bots[s])
        action = bots[s].choose(view, s)
        assert action in view.actions or action['type'] == 'offer_trade'
        game.apply(s, action)  # the engine refuses anything illegal
    assert game.winner is not None
    records = bots[0].diagnostics
    assert records and {d['phase'] for d in records} == {'setup_settlement', 'actions'}
    assert all(d['simulations'] == ps.MIN_SIMULATIONS and not d['fallback'] for d in records)
    assert max(d['max_depth'] for d in records) >= 2
    for d in records:
        assert set(d) >= {'reason', 'seconds', 'rollouts', 'truncated'}  # evaluator fields


@needs_weights
def test_a_first_level_leaf_is_valued_exactly_like_pv2_tiles():
    from catan_ai.pv_search import afterstate_encoding
    game = _game_at_our_turn_plan(2950023)
    planner = make_policy('planner_pv2_puct', 3)
    view = Observation.for_policy(game, 0, planner)
    env = ps.CatanEnv(planner, view, 0, planner.turn_candidates(view, 0))
    state = env.sample(0)
    end_now = next(c for c in env.candidates(state) if c[0] == '[]')
    _, value = env.play(state, end_now[2], expand=True)
    x, won = afterstate_encoding(planner, view, 0, [])
    assert not won
    assert value == pytest.approx(1 / (1 + math.exp(-planner._logits([x])[0])), abs=1e-12)
