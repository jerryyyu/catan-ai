"""Board row 4, integration: the served bot with the learned win model as its build-search leaf.

``planner_value_leaf`` is ``planner_offer_accept`` whose turn-search leaf is the
log-odds of the frozen logistic model from ``runs/value-gate-v1.json`` (offline
gate PASS: test AUC .703 vs .652 shared score and .588 served leaf). Robber,
steal, discard, development-card, trade-answer and offer decisions are unchanged.

The model was fit on positions at the start of the seat's own turn; the search
evaluates positions after its own builds, before the next roll. That shift is a
known limitation of this integration, not corrected here. The checkpoint is
bound to its gate receipt and to the feature producers' source hashes and fails
closed on any mismatch.
"""
import hashlib
import json
from pathlib import Path

from .experiments import source_files
from .observation import Observation
from .unified_planner import AcceptanceOfferPlanner
from .value_model import value_features

CHECKPOINT = Path(__file__).resolve().parents[2] / 'checkpoints' / 'value_leaf_v1.json'
CHECKPOINT_SHA256 = '59979aefe85af74d2c74331fa51e95ca0bad407709bc4ac8bb1ddfc0f2f08521'
WIN = 1e5


def load_value_leaf(path=CHECKPOINT):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('value-leaf checkpoint does not match the frozen model')
    data = json.loads(raw)
    current = source_files()
    if any(current.get(name) != sha for name, sha in data['producer_source'].items()):
        raise ValueError('value-leaf feature producers changed since the gate')
    model = data['model']
    if not (len(model['weights']) == len(model['means']) == len(model['scales']) == 40):
        raise ValueError('invalid value-leaf model')
    return model


class ValueLeafPlanner(AcceptanceOfferPlanner):
    """``planner_offer_accept`` with the learned win log-odds as the build-search leaf."""

    def __init__(self, rng=None):
        super().__init__(rng)
        self.value_model = load_value_leaf()

    def _logit(self, game, seat):
        leaves = [self._value(game, s) for s in range(len(game.players))]
        view = Observation.from_game(game, seat, include_events=False)
        x = value_features(view) + [leaves[seat] / 100,
                                    max(leaves[s] for s in range(len(leaves)) if s != seat) / 100]
        m = self.value_model
        return m['bias'] + sum(w * (v - mu) / s for w, v, mu, s in zip(m['weights'], x, m['means'], m['scales']))

    def _leaf(self, game, seat):
        if game.players[seat].points >= 10:
            return WIN
        return self._logit(game, seat)


CHECKPOINT_V2 = CHECKPOINT.with_name('value_leaf_v2.json')
CHECKPOINT_V2_SHA256 = '955ca605d43e5b127423a6024ece18eb89f49d67ce7330deedef2613292c3a04'


class ScaledValueLeafPlanner(ValueLeafPlanner):
    """v2: the same frozen model, its log-odds mapped onto the served leaf's units.

    ``leaf = leaf_mean + slope * (logit - logit_mean)`` with mean/SD matching fitted
    on the train corpus only (``scripts/fit_value_leaf_scale.py``), so the search's
    fixed purchase bonus keeps the relative weight it has for the served leaf.
    """

    def __init__(self, rng=None):
        super().__init__(rng)
        raw = CHECKPOINT_V2.read_bytes()
        if hashlib.sha256(raw).hexdigest() != CHECKPOINT_V2_SHA256:
            raise ValueError('value-leaf v2 checkpoint does not match the frozen scale')
        data = json.loads(raw)
        if data['base_checkpoint_sha256'] != CHECKPOINT_SHA256 or data['model'] != self.value_model:
            raise ValueError('value-leaf v2 is not built on the frozen v1 model')
        self.scale = data['scale']

    def _leaf(self, game, seat):
        if game.players[seat].points >= 10:
            return WIN
        s = self.scale
        return s['leaf_mean'] + s['slope'] * (self._logit(game, seat) - s['logit_mean'])
