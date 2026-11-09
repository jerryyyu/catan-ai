"""Turn-exact search over the served tiles value (``planner_pv2_exact``; board #1 "Turn-exact search").

``planner_pv2_tiles`` scores five end-of-turn candidates: ending now plus the served planner's four
best sequences by first move (a beam search ranked by the economic leaf). This planner scores the
same five first, exactly as ``planner_pv2_tiles`` does (same afterstate features, same tiles value),
then every further own-turn afterstate reachable with builds, bank/port trades and one development
purchase (a purchase ends a sequence, as in the served search), enumerated breadth-first on the
chooser's card-counted world and deduplicated by the resulting own position. It plays the first move
of the highest-valued afterstate and replans on every call, like ``planner_pv2_tiles``.

Budgets are counts, not clocks, so play is deterministic: at most ``MAX_NODES`` expansions and
``MAX_STATES`` scored afterstates per decision (breadth-first, so shallower sequences come first).
Because the five served candidates are scored first with identical features and a candidate replaces
the incumbent only when strictly better, this planner never picks an afterstate the tiles value rates
below ``planner_pv2_tiles``'s pick; every difference is a strictly higher-valued afterstate (including
immediate wins the beam search missed).
"""
from collections import deque
from copy import deepcopy
import json
import math

from .pv_search import PVSearchTilesPlanner, REPLACEABLE, afterstate_encoding, candidate_sequences, encode_afterstate
from .unified_planner import AcceptanceOfferPlanner

EXPAND = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development')


def _own_key(game, seat, bought):
    p = game.players[seat]
    return (tuple(sorted((str(r), n) for r, n in p.resources.items())), tuple(sorted(p.settlements)),
            tuple(sorted(p.cities)), tuple(sorted(tuple(sorted(e)) for e in p.roads)), bought)


def turn_sequences(planner, view, seat, max_nodes, max_states, worlds=None):
    """Own-turn sequences reaching distinct afterstates, breadth-first; (sequences, nodes, capped).

    With ``worlds`` (a dict), each returned sequence's afterstate world and purchase flag are stored
    under its JSON key, so the caller can encode it without replaying the sequence again."""
    root = planner._world(view)
    seen = {_own_key(root, seat, False)}
    out, queue, nodes, capped = [], deque([(root, [])]), 0, False
    while queue and not capped:
        game, seq = queue.popleft()
        for action in game.legal_actions(seat):
            if action['type'] not in EXPAND:
                continue
            if nodes >= max_nodes or len(out) >= max_states:
                capped = True
                break
            nodes += 1
            child = deepcopy(game, {id(game.board): game.board})
            bought = action['type'] == 'buy_development'
            if bought:
                if not child.players[seat].can_afford('development'):
                    continue
                child.players[seat].pay('development')
            else:
                child.apply(seat, action)
            key = _own_key(child, seat, bought)
            if key in seen:
                continue
            seen.add(key)
            out.append(seq + [action])
            if worlds is not None:
                worlds[json.dumps(out[-1], sort_keys=True)] = (child, bought)
            if not bought and child.winner is None:
                queue.append((child, seq + [action]))
    return out, nodes, capped


class TurnExactPlanner(PVSearchTilesPlanner):
    """``planner_pv2_tiles`` with every reachable own-turn afterstate scored, not just five."""

    MAX_NODES = 5000
    MAX_STATES = 1200

    def __init__(self, rng=None):
        super().__init__(rng)
        self.last_search = None
        self._cache_key, self._cache = None, {}

    def _cached_logits(self, turn_key, rows):
        """Logits with a within-turn cache keyed by the exact feature row; later calls in a turn
        re-score mostly the same afterstates. Misses are scored in one batch."""
        if turn_key != self._cache_key:
            self._cache_key, self._cache = turn_key, {}
        keys = [tuple(r) for r in rows]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            fresh = self._logits([rows[i] for i in missing])
            for i, z in zip(missing, fresh):
                self._cache[keys[i]] = z
        return [self._cache[k] for k in keys]

    def choose(self, view, seat):
        before = self.last_offer_turn
        served = AcceptanceOfferPlanner.choose(self, view, seat)
        if (view.phase != 'actions' or served['type'] in ('offer_trade', 'play_development')
                or served['type'] not in REPLACEABLE or not any(a['type'] == 'end_turn' for a in view.actions)):
            return served
        pv = candidate_sequences(self, view, seat, self.K)
        built = {}
        extra, nodes, capped = turn_sequences(self, view, seat, self.MAX_NODES, self.MAX_STATES, built)
        # The card-counted world estimates opponents' hands and so the bank's supply; only the real
        # legal set may supply the move actually played (exact-screen-v1 aborted on an illegal bank
        # trade the world allowed). Deeper steps are a plan, replanned against the real game.
        legal = view.legal_actions(seat)
        dropped = sum(1 for s in extra if s[0] not in legal)
        listed = {json.dumps(s, sort_keys=True) for s in pv}
        seqs = pv + [s for s in extra if s[0] in legal and json.dumps(s, sort_keys=True) not in listed]
        scored = []
        for seq in seqs:
            reuse = built.get(json.dumps(seq, sort_keys=True))
            # an enumerated afterstate is the same world the replay would rebuild (same root, legal moves)
            encoded = (encode_afterstate(reuse[0], reuse[1], view, seat) if reuse is not None
                       else afterstate_encoding(self, view, seat, seq))
            if encoded is not None:
                scored.append((seq, *encoded))
        logits = self._cached_logits((view.turn, seat), [x for _, x, won in scored if not won]) if scored else []
        best, pick, it = -math.inf, None, iter(logits)
        for seq, _, won in scored:
            value = math.inf if won else next(it)
            if value > best:
                best, pick = value, seq
        self.last_offer_turn = before
        self.last_search = {'pv_candidates': len(pv), 'scored': len(scored), 'nodes': nodes, 'capped': capped,
                            'dropped_illegal_first': dropped,
                            'from_pv': pick is not None and json.dumps(pick, sort_keys=True) in listed}
        if pick is None:
            return served
        if not pick:  # end the turn: the served end-of-turn path, as planner_pv2_tiles does
            if served['type'] == 'end_turn':
                return served
            legal = view.legal_actions(seat)
            action = (self._fallback(view, seat) if any(a['type'] == 'play_development' for a in legal)
                      else {'type': 'end_turn'})
            return self._end_of_turn_offer(view, seat, action)
        if pick[0] not in legal:  # defensive: a served candidate is always legal, extras were filtered
            return served
        return pick[0]
