"""Frozen outcome-value augmentation of the existing economic planner."""
import hashlib
import json
import math
from pathlib import Path

from .observation import Observation
from .planner import ProductionRobberPlanner
from .value_model import value_features

MODEL_SHA = 'b55eab9866dce201c28b67b76b5e3a71ad0250854b56e35f125c9baff5b56a0f'
MODEL_PATH = Path(__file__).resolve().parents[2] / 'runs/outcome-value-v1/model.json'


def load_value_model(path=MODEL_PATH):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA:
        raise ValueError('outcome-value checkpoint does not match the frozen model')
    data = json.loads(raw)
    model = data['value']
    for key in ('weights', 'means', 'scales'):
        if len(model[key]) != 38 or any(not math.isfinite(x) for x in model[key]):
            raise ValueError('invalid outcome-value vector')
    if any(x <= 0 for x in model['scales']):
        raise ValueError('invalid outcome-value scales')
    for name in ('value_model.py', 'learning.py', 'strategic.py', '_engine_py.py'):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != data['source'][name]:
            raise ValueError('outcome-value feature producer changed')
    return model


class ValuePlanner(ProductionRobberPlanner):
    def __init__(self, rng):
        super().__init__(rng)
        self.model = load_value_model()
        self.model_hash = MODEL_SHA
        self.load_error = None

    def _value(self, game, seat):
        base = super()._value(game, seat)
        if game.phase.startswith('setup') or game.players[seat].points >= 10:
            return base
        features = value_features(Observation.from_game(game, seat, include_events=False))
        logit = sum(w * (x - m) / s for w, x, m, s in zip(
            self.model['weights'], features, self.model['means'], self.model['scales'], strict=True))
        return base + 20 * logit


class ValueDrawPlanner(ValuePlanner):
    @property
    def purchase_value(self):
        # The fixed deck prior estimates an unknown draw; it is not knowledge
        # of remaining identities or an exact posterior over the depleted deck.
        weights, scales = self.model['weights'], self.model['scales']
        expected = weights[6] / scales[6] / 25
        expected += sum(count / 25 * weights[index] / scales[index]
                        for index, count in enumerate((5, 14, 2, 2, 2), 21))
        return 8 + 20 * expected
