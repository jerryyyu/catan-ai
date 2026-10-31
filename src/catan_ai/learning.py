"""Small observation-only linear policy-ranking and win-value baseline."""
import hashlib
import json
import math
from pathlib import Path
import random

from .engine import COSTS, Resource
from .strategic import production
from .tactical import TacticalBot

SCHEMA = 1
KINDS = ('build_city', 'build_settlement', 'build_road', 'bank_trade',
         'buy_development', 'play_development', 'end_turn', 'offer_trade')
CARDS = ('knight', 'year_of_plenty', 'road_building', 'monopoly')


def state_features(view):
    p = view.players[view.seat]
    buildings = {v: 1 for v in p.settlements}
    buildings.update({v: 2 for v in p.cities})
    income = production(view, buildings)
    return [1., p.public_points / 10, len(p.roads) / 15, len(p.settlements) / 5,
            len(p.cities) / 4, p.knights / 14, len(p.development) / 25,
            min(view.turn, 400) / 400] + [p.resources[r] / 19 for r in Resource] + [
                income[r] / 30 for r in Resource] + [
                view.players[(view.seat + i) % 4].public_points / 10 for i in (1, 2, 3)]


def action_features(view, action):
    p = view.players[view.seat]
    kind = action['type']
    cost = COSTS.get(kind.removeprefix('build_'), {})
    if kind == 'buy_development':
        cost = COSTS['development']
    after = dict(p.resources)
    for r, n in cost.items():
        after[r] -= n
    if kind == 'bank_trade':
        give, receive = Resource(action['give']), Resource(action['receive'])
        after[give] -= view.rates[give]
        after[receive] += 1
    vertices = [action['vertex']] if 'vertex' in action else action.get('edge', [])
    income = production(view, {v: 1 for v in vertices})
    return ([float(kind == k) for k in KINDS] +
            [float(action.get('card') == c) for c in CARDS] +
            [after[r] / 19 for r in Resource] + [income[r] / 30 for r in Resource] +
            [sum(max(0, n - after[r]) for r, n in COSTS[target].items()) / 5
             for target in ('city', 'settlement', 'road', 'development')])


def dot(weights, features):
    if len(weights) != len(features):
        raise ValueError('feature dimension mismatch')
    return sum(w * x for w, x in zip(weights, features))


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


def probabilities(scores):
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def load_model(path):
    raw = Path(path).read_bytes()
    model = json.loads(raw)
    if model.get('schema') != SCHEMA:
        raise ValueError('unknown model schema')
    for key, size in (('policy_weights', 26), ('value_weights', 21)):
        if len(model[key]) != size or any(type(w) not in (int, float) or not math.isfinite(w) for w in model[key]):
            raise ValueError('invalid model weights')
    return model, hashlib.sha256(raw).hexdigest()


class LearnedBot(TacticalBot):
    def __init__(self, seed, path):
        super().__init__(random.Random(seed))
        self.model, self.model_hash, self.load_error = None, None, None
        self.seen_turn, self.seen = None, set()
        try:
            self.model, self.model_hash = load_model(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.load_error = str(exc)

    def win_probability(self, view):
        if self.model is None:
            return None
        return sigmoid(dot(self.model['value_weights'], state_features(view)))

    def choose(self, view, seat):
        if self.model is None or view.phase != 'actions' or not view.actions:
            return super().choose(view, seat)
        if self.seen_turn != view.turn:
            self.seen_turn, self.seen = view.turn, set()
        signature = json.dumps([state_features(view), list(view.actions)], sort_keys=True)
        if signature in self.seen:
            return super().choose(view, seat)
        self.seen.add(signature)
        # Enumerating observed legal actions is the action mask.
        scores = [dot(self.model['policy_weights'], action_features(view, a)) for a in view.actions]
        return dict(view.actions[max(range(len(scores)), key=scores.__getitem__)])
