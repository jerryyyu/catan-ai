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


SERVED_CHECKPOINT = Path(__file__).resolve().parents[2] / 'checkpoints' / 'planner_parameter.json'


def load_parameters(path=None):
    """Load a source-pinned checkpoint.

    The path is, in order: the argument, ``CATAN_PARAMETER_MODEL``, then the
    served checkpoint committed under ``checkpoints/``. Any mismatch fails closed.
    """
    path = Path(path or os.environ.get('CATAN_PARAMETER_MODEL') or SERVED_CHECKPOINT).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get('weights'), list) \
            or not isinstance(data.get('source'), dict) or 'schema' not in data:
        raise ValueError('invalid economic parameter checkpoint structure')
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


def served_status():
    """Return (available, reason) for the checkpoint new games would load."""
    try:
        load_parameters()
    except (KeyError, OSError, ValueError) as exc:
        return False, type(exc).__name__ + ': ' + str(exc)
    return True, None


def compatibility_games(plan):
    """Play a checkpoint's seeded compatibility games and return full traces.

    Each game puts the fitted planner in one seat against ``plan['opponents']``.
    The record shape matches the v13 evaluation binding's trace file, so a
    rebind can be compared with the games the evaluated snapshot played.
    """
    from .bots import make_policy
    from .engine import Game
    from .observation import Observation
    results = []
    for seed in plan['seeds']:
        for focal in plan['seats']:
            names = list(plan['opponents'])
            names.insert(focal, 'planner_parameter')
            game = Game.new([f'{name}-{seat}' for seat, name in enumerate(names)], seed)
            policies = [make_policy(name, seed * 17 + seat + 1) for seat, name in enumerate(names)]
            trace = []
            while game.winner is None and game.turn < plan['max_turns'] and len(trace) < 50000:
                seat = game.current
                action = policies[seat].choose(Observation.for_policy(game, seat, policies[seat]), seat)
                game.apply(seat, action)
                trace.append({'seat': seat, 'action': action})
            results.append({'seed': seed, 'seat': focal, 'winner': game.winner, 'turn': game.turn,
                            'points': [p.points for p in game.players], 'trace': trace,
                            'final_rng': repr(game.rng.getstate())})
    return results


def trace_digest(games):
    return hashlib.sha256(json.dumps(games, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compatibility_digest(plan):
    return trace_digest(compatibility_games(plan))
