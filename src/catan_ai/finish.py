"""Bounded deterministic winning sequences, with unchanged economic fallback."""
from copy import deepcopy

from .chain_planner import bank_floor, supported
from .engine import COSTS, Resource
from .planner import ProductionRobberPlanner
from .tactical import public_projection


def winning_plan(view, seat, depth=6, node_cap=768):
    player = view.players[seat]
    if (view.phase not in ('roll', 'actions', 'free_roads')
            or player.public_points + player.development.count('victory_point') < 7):
        return []
    root = public_projection(view)
    nodes = 0
    seen = {}
    priority = {'build_city': 0, 'build_settlement': 1, 'play_development': 2,
                'bank_trade': 3, 'build_road': 4}

    def visit(game, actions, floor, path):
        nonlocal nodes
        def order(action):
            usefulness = 0
            if action['type'] == 'bank_trade':
                hand = game.players[seat].resources
                give, receive = Resource(action['give']), Resource(action['receive'])
                usefulness = max(weight * int(hand[receive] < cost.get(receive, 0))
                                 * int(hand[give] - game.trade_rate(seat, give) >= cost.get(give, 0))
                                 for weight, cost in ((2, COSTS['city']), (1, COSTS['settlement'])))
            return priority.get(action['type'], 9), -usefulness

        for action in sorted(actions, key=order):
            kind = action['type']
            knight = kind == 'play_development' and action['card'] == 'knight'
            if kind == 'buy_development' or not (knight or supported(action, floor)):
                continue
            if nodes >= node_cap:
                return []
            nodes += 1
            child = deepcopy(game, {id(game.board): game.board})
            child.apply(seat, action)
            plan = path + [action]
            if child.winner == seat:
                return plan
            if len(plan) == depth or child.phase not in ('actions', 'free_roads'):
                continue
            p = child.players[seat]
            remaining = {r: floor[r] + game.players[seat].resources[r] - p.resources[r] for r in floor}
            key = (tuple(p.resources.values()), frozenset(p.roads), frozenset(p.settlements),
                   frozenset(p.cities), tuple(p.development), p.knights, child.phase,
                   child.free_roads, child.played_development, tuple(remaining.values()))
            if seen.get(key, depth + 1) <= len(plan):
                continue
            seen[key] = len(plan)
            result = visit(child, child.legal_actions(seat), remaining, plan)
            if result:
                return result
        return []

    return visit(root, view.actions, bank_floor(view), [])


class FinishPlanner(ProductionRobberPlanner):
    def choose(self, view, seat):
        plan = winning_plan(view, seat)
        return plan[0] if plan else super().choose(view, seat)
