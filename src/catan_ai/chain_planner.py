"""Multi-acquisition turn plans with conservative public bank availability."""
from copy import deepcopy
from itertools import zip_longest
import math

from .engine import Resource
from .planner import ProductionRobberPlanner
from .tactical import public_projection


def bank_floor(view):
    other_cards = sum(p.resource_count for s, p in enumerate(view.players) if s != view.seat)
    floor = {r: max(0, 19 - view.players[view.seat].resources[r] - other_cards) for r in Resource}
    for action in view.actions:
        if action['type'] == 'bank_trade':
            r = Resource(action['receive'])
            floor[r] = max(floor[r], 1)
        elif action['type'] == 'play_development' and action['card'] == 'year_of_plenty':
            for r in Resource:
                floor[r] = max(floor[r], action['resources'].count(r.value))
    return floor


def supported(action, floor):
    if action['type'] == 'bank_trade':
        return floor[Resource(action['receive'])] >= 1
    if action['type'] == 'play_development' and action['card'] == 'year_of_plenty':
        return all(action['resources'].count(r.value) <= floor[r] for r in Resource)
    return action['type'] in ('build_city', 'build_settlement', 'build_road', 'buy_development') or (
        action['type'] == 'play_development' and action['card'] == 'road_building')


class ChainPlanner(ProductionRobberPlanner):
    def choose(self, view, seat):
        if view.phase not in ('actions', 'free_roads'):
            return super().choose(view, seat)
        root = public_projection(view)
        best = next((a for a in view.actions if a['type'] == 'end_turn'), None)
        best_score = self._value(root, seat) if best else -math.inf
        frontier = [(root, None, bank_floor(view), view.actions)]
        for depth in range(5):
            successors = []
            seen = set()
            groups = [[(game, first, floor, action) for action in actions if supported(action, floor)
                       and (depth == 0 or action['type'] != 'buy_development')]
                      for game, first, floor, actions in frontier]
            nodes = 0
            for batch in zip_longest(*groups):
                for row in batch:
                    if row is None:
                        continue
                    if nodes >= 96:
                        break
                    nodes += 1
                    game, first, floor, action = row
                    child = deepcopy(game, {id(game.board): game.board})
                    buy = action['type'] == 'buy_development'
                    if buy:
                        child.players[seat].pay('development')
                    else:
                        child.apply(seat, action)
                    score = self._value(child, seat) + (self.purchase_value if buy else 0) - .01 * (depth + 1)
                    first = first or action
                    if score > best_score:
                        best, best_score = first, score
                    if buy or child.winner is not None:
                        continue
                    player = child.players[seat]
                    key = (tuple(player.resources.values()), frozenset(player.roads),
                           frozenset(player.settlements), frozenset(player.cities),
                           tuple(player.development), child.phase, child.free_roads, child.played_development)
                    if key in seen:
                        continue
                    seen.add(key)
                    remaining = {r: floor[r] + game.players[seat].resources[r] - player.resources[r] for r in Resource}
                    successors.append((score, child, first, remaining))
                if nodes >= 96:
                    break
            successors.sort(key=lambda row: row[0], reverse=True)
            frontier = [(game, first, floor, game.legal_actions(seat))
                        for _, game, first, floor in successors[:8]]
            if not frontier:
                break
        if best is None or best['type'] == 'end_turn' and any(a['type'] == 'play_development' for a in view.actions):
            return super().choose(view, seat)
        return best
