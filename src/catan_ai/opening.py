"""Bounded setup continuations using only the observed public board."""
from copy import deepcopy

from .bots import HeuristicBot
from .engine import Resource
from .planner import ProductionRobberPlanner
from .tactical import public_projection


class OpeningPlanner(ProductionRobberPlanner):
    opening_model = 'economic'

    def _opening_value(self, game, seat):
        return self._value(game, seat)

    def _child(self, game, action):
        child = deepcopy(game, {id(game.board): game.board})
        child.apply(game.current, action)
        return child

    def _ranked(self, game, seat, model):
        rows = []
        for action in game.legal_actions(seat):
            child = self._child(game, action)
            if game.phase == 'setup_road':
                value = self._geometry(child, seat)
            elif model == 'production':
                value = HeuristicBot._vertex_score(game, action['vertex'])
            elif model == 'purchase':
                value = self._opening_value(child, seat)
            else:
                value = self._value(child, seat)
            rows.append((value, action, child))
        return sorted(rows, key=lambda row: row[0], reverse=True)

    def choose(self, view, seat):
        if not view.phase.startswith('setup'):
            return super().choose(view, seat)
        root = public_projection(view)
        root.setup_index = view.setup_index
        root.pending_setup_vertex = view.pending_setup_vertex
        # Setup resources are determined by placements, without hidden draws.
        ranked = self._ranked(root, seat, self.opening_model)
        candidates = ranked[:8] if view.phase == 'setup_settlement' else ranked
        best_action, best_score = candidates[0][1], float('-inf')
        for _, action, child in candidates:
            values = []
            for opponent_model in ('economic', 'production'):
                game = child
                while game.phase.startswith('setup'):
                    actor = game.current
                    model = self.opening_model if actor == seat else opponent_model
                    game = self._ranked(game, actor, model)[0][2]
                values.append(self._opening_value(game, seat))
            score = sum(values) / len(values)
            if score > best_score:
                best_action, best_score = action, score
        return best_action


def purchase_horizon(hand, income, rates, cost):
    """Fluid production horizon in dice rolls, allowing fractional bank trades.

    This is a deterministic affordability proxy, not expected stochastic hitting
    time: it excludes robber losses, finite bank contention and player trades.
    """
    def affordable(rolls):
        surplus = deficit = 0.0
        for resource in Resource:
            remaining = hand[resource] + income[resource] * rolls / 36 - cost.get(resource, 0)
            if remaining >= 0:
                surplus += remaining / rates[resource]
            else:
                deficit -= remaining
        return surplus >= deficit
    if affordable(0):
        return 0.0
    if not affordable(128):
        return float('inf')
    lo, hi = 0.0, 128.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if affordable(mid):
            hi = mid
        else:
            lo = mid
    return hi


class PurchaseOpeningPlanner(OpeningPlanner):
    opening_model = 'purchase'

    def _opening_value(self, game, seat):
        income = self._income(game, seat)
        hand = game.players[seat].resources
        rates = {r: game.trade_rate(seat, r) for r in Resource}
        city = purchase_horizon(hand, income, rates, {Resource.ORE: 3, Resource.GRAIN: 2})
        expansion = purchase_horizon(hand, income, rates, {
            Resource.BRICK: 2, Resource.LUMBER: 2, Resource.WOOL: 1, Resource.GRAIN: 1})
        # One additional road plus a settlement models the nearest expansion;
        # geometry separately discounts positions with poor reachable sites.
        return self._value(game, seat) + 30 / (1 + city / 4) + 20 / (1 + expansion / 4)
