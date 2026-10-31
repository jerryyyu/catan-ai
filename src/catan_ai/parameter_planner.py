"""Source-pinned economic weights trained against complete-game outcomes."""
import hashlib
import json
import math
import os
from pathlib import Path

from .experiments import source_files
from .planner import ProductionRobberPlanner, WEIGHTS
from .engine import Resource


DEFAULT = [12., .55, 1., 1., .5, 1., 8., 2.]
BOUNDS = [(6., 30.), (.2, 1.2), (.2, 2.), (.1, 3.), (.1, 2.), (.2, 3.), (2., 18.), (.2, 5.)]


def load_parameters(path=None):
    path = Path(path or os.environ['CATAN_PARAMETER_MODEL']).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    weights = data['weights']
    if (data['schema'] != 1 or len(weights) != len(BOUNDS)
            or any(type(x) not in (int, float) or not math.isfinite(x) or not lo <= x <= hi
                   for x, (lo, hi) in zip(weights, BOUNDS))):
        raise ValueError('invalid economic parameters')
    if data['source'] != source_files():
        raise ValueError('economic parameter source mismatch')
    return weights, hashlib.sha256(raw).hexdigest(), path


class ParameterPlanner(ProductionRobberPlanner):
    def __init__(self, rng):
        super().__init__(rng)
        self.weights, self.model_hash, _ = load_parameters()
        self.load_error = None
        self.purchase_value = self.weights[6]

    def _value(self, game, seat):
        player = game.players[seat]
        if player.points >= 10:
            return 100000.
        w = self.weights
        income = self._income(game, seat)
        production = sum(WEIGHTS[r] * (n + w[7] * math.sqrt(n)) for r, n in income.items())
        hand = sum(WEIGHTS[r] * (1.6 * min(n, 3) + .45 * max(n - 3, 0))
                   for r, n in player.resources.items())
        discard = .5 * max(sum(player.resources.values()) - 7, 0)
        ports = sum(max(income[r] - 5, 0) * (4 / game.trade_rate(seat, r) - 1) for r in Resource)
        development = sum(6 if card == 'victory_point' else 3 for card, _ in player.development)
        return (w[0] * player.points + w[1] * production + w[2] * hand - discard
                + w[3] * self._geometry(game, seat) + w[4] * ports + w[5] * development)
