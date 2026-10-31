"""Observation-only win-value features and a small linear model."""
import math

from .engine import Resource, COSTS
from .learning import state_features


def value_features(view):
    base = state_features(view)
    player = view.players[view.seat]
    cards = [player.development.count(card) for card in
             ('victory_point', 'knight', 'monopoly', 'year_of_plenty', 'road_building')]
    deficits = [sum(max(0, amount - player.resources[r]) for r, amount in COSTS[kind].items())
                for kind in ('city', 'settlement', 'road', 'development')]
    opponent_income = []
    for offset in (1, 2, 3):
        p = view.players[(view.seat + offset) % 4]
        income = sum((6 - abs(7 - tile.number)) * sum((v in p.settlements) + 2 * (v in p.cities) for v in tile.vertices)
                     for tile in view.board.tiles if tile.number is not None)
        opponent_income.append(income / 30)
    return base + cards + deficits + [view.rates[r] / 4 for r in Resource] + opponent_income


def baseline_features(view):
    x = state_features(view)
    return [1., x[1], max(x[18:21]), sum(x[13:18]), view.turn / 100,
            view.players[view.seat].development.count('victory_point')]


def predict(weights, means, scales, features):
    score = sum(w * (x - m) / s for w, x, m, s in zip(weights, features, means, scales, strict=True))
    return 1 / (1 + math.exp(-max(-40, min(40, score))))
