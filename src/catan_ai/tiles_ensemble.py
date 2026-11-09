"""Five-seed tiles ensemble (``planner_pv2_tiles_ens5``; experiments/tiles-tta-ens.json arm ``ens5``).

The offline gate (runs/tiles-ens-v1/gate.json on ultra; board #1 comment 6748388722) passed the mean
P(I win) of five arch-bakeoff ``tiles`` models: the served weights (seed 0) plus seeds 1-4 trained with
experiments/arch-bakeoff.json changed only in ``common.seed``. This module serves that arm in NumPy:
each member is a ``TilesValueNet`` (its own SHA-256 and torch parity checks), every member's source
checkpoint is pinned to the hash the gate scored, and the ensemble logit is the logit of the members'
mean probability, so the PV comparison ranks afterstates exactly as the gate's ``ens5`` scorer did.

Weights are local artifacts (``checkpoints/value_tiles_ens_s{1..4}.npz`` plus the served
``value_tiles_v1.npz``); only their metadata is committed.
"""
import hashlib
from pathlib import Path

from .pv_search import PVSearchTilesPlanner
from .unified_planner import AcceptanceOfferPlanner

CHECKPOINTS = Path(__file__).resolve().parents[2] / 'checkpoints'
# (weights stem, source .pt SHA-256 the gate scored), in the gate's ensemble order.
ENS5 = (
    ('value_tiles_v1', 'fb239324ad2195b77965f69aa28e7f8dcf3caa756e84d1892ba5f804a33972ae'),
    ('value_tiles_ens_s1', '7c370b227f2e8cbdca40914f3b1ae8c17214cc25ba6249b5b26450d67ac7c232'),
    ('value_tiles_ens_s2', 'fe2710ac969daa25df05a37d632f2a864685cc90afd53a3077f622bef267b194'),
    ('value_tiles_ens_s3', 'f780db65298c56cd181ba53cf017f68ff875f7e5d9d4b26b7d645abff916e3df'),
    ('value_tiles_ens_s4', '875a02fda672a515425e9d9b06cac3b861e14c45e9ae3052fd310a27385ced08'),
)


class TilesEnsembleNet:
    """Mean-probability ensemble of exported tiles transformers; ``logits`` returns logit(mean p)."""

    def __init__(self, members=ENS5, root=CHECKPOINTS):
        from .tiles_value import TilesValueNet
        import numpy as np
        self.np = np
        self.nets = []
        for stem, pt_sha in members:
            net = TilesValueNet(Path(root) / stem)
            if net.meta.get('pt_sha256') != pt_sha:
                raise ValueError(f'{stem}: exported from {net.meta.get("pt_sha256")}, the gate scored {pt_sha}')
            self.nets.append(net)
        joined = ''.join(n.meta['npz_sha256'] for n in self.nets)
        self.meta = {'npz_sha256': hashlib.sha256(joined.encode()).hexdigest(),
                     'members': [{'stem': s, 'pt_sha256': p, 'npz_sha256': n.meta['npz_sha256']}
                                 for (s, p), n in zip(members, self.nets)]}

    def logits(self, rows):
        np = self.np
        p = np.mean([1 / (1 + np.exp(-np.asarray(n.logits(rows), dtype=np.float64))) for n in self.nets], axis=0)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))


class PVSearchTilesEnsPlanner(PVSearchTilesPlanner):
    """``planner_pv2_tiles`` with the five-seed ensemble as its value; candidates and integration unchanged."""

    def __init__(self, rng=None):
        AcceptanceOfferPlanner.__init__(self, rng)
        self.net = TilesEnsembleNet()
