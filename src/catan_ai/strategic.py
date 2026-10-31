"""Resource-aware placement candidate; constants are hypotheses tested in DEV."""

import math

from .bots import HeuristicBot
from .engine import Resource
from .observation import Observation


def production(game: Observation, buildings: dict[int, int]) -> dict[Resource, float]:
    income = {r: 0.0 for r in Resource}
    for tile in game.board.tiles:
        if isinstance(tile.resource, Resource):
            pips = 6 - abs(7 - tile.number)
            income[tile.resource] += pips * sum(buildings.get(v, 0) for v in tile.vertices)
    return income


def placement_score(game: Observation, seat: int, vertex: int, city: bool = False) -> float:
    player = game.players[seat]
    buildings = {v: 1 for v in player.settlements}
    buildings.update({v: 2 for v in player.cities})
    buildings[vertex] = 2 if city else 1
    income = production(game, buildings)
    weights = {Resource.BRICK: 1.0, Resource.LUMBER: 1.0, Resource.WOOL: .8,
               Resource.GRAIN: 1.4, Resource.ORE: 1.5}
    score = sum(weights[r] * math.log1p(n) for r, n in income.items())
    score += .15 * sum(income.values())
    expansion = min(income[Resource.BRICK], income[Resource.LUMBER], income[Resource.GRAIN], income[Resource.WOOL])
    cities = min(income[Resource.ORE] / 3, income[Resource.GRAIN] / 2)
    development = min(income[Resource.ORE], income[Resource.GRAIN], income[Resource.WOOL])
    score += .5 * max(expansion, cities + .5 * development)
    for endpoints, kind in game.board.ports:
        if any(v in buildings for v in endpoints):
            score += .25 * (max(income.values()) if kind is None else income[kind])
    return score


class StrategicBot(HeuristicBot):
    def choose(self, game: Observation, seat: int) -> dict:
        legal = game.legal_actions(seat)
        cities = [a for a in legal if a['type'] == 'build_city']
        settlements = [a for a in legal if a['type'] == 'build_settlement']
        # Preserve the baseline's development-card priority outside setup.
        if not any(a['type'] == 'play_development' for a in legal):
            if cities:
                return max(cities, key=lambda a: placement_score(game, seat, a['vertex'], city=True))
            if settlements:
                return max(settlements, key=lambda a: placement_score(game, seat, a['vertex']))
        return super().choose(game, seat)
