"""DEV candidates from experiments/value-arms.json: ``planner_pv2_tiles_va_<arm>``.

Each candidate is ``planner_pv2_tiles_buyfix`` (served PV search with the red-team purchase-scoring fixes, PR #59;
bundled per the owner's decision, board #1 comment 6756308802) with its value transformer replaced by the
value-arms checkpoint ``<arm>``, exported for NumPy serving (``scripts/export_tiles_npz.py``). Each export is
pinned to the .pt SHA-256 that ``scripts/value_arms_gate.py`` scored. Weights are local artifacts
(``checkpoints/value_tiles_va_<arm>.npz``); only the meta files are committed.
"""
from pathlib import Path

from .purchase_scoring import PVSearchTilesBuyFixPlanner
from .unified_planner import AcceptanceOfferPlanner

CHECKPOINTS = Path(__file__).resolve().parents[2] / 'checkpoints'
# arm -> the .pt SHA-256 the offline gate scored (runs/value-arms-v1/<arm>-gate.json on ultra)
ARMS = {
    't9': 'db4bc28e01b37dd74f963ff3b8cddd7789a0f22e8ace1a0bc73f99541cc60c12',
    't8': 'ad8038358820e66d022dedf4b27a28b4bd450d2cf4f2c3aae23cbb424a4d622a',
}


class ValueArmPlanner(PVSearchTilesBuyFixPlanner):
    """``planner_pv2_tiles_buyfix`` scored by one value-arms checkpoint."""

    ARM = None

    def __init__(self, rng=None):
        from .tiles_value import TilesValueNet
        AcceptanceOfferPlanner.__init__(self, rng)
        net = TilesValueNet(CHECKPOINTS / f'value_tiles_va_{self.ARM}')
        if net.meta.get('pt_sha256') != ARMS[self.ARM] or net.meta.get('value_arm') != self.ARM:
            raise ValueError(f'value_tiles_va_{self.ARM}: exported from {net.meta.get("pt_sha256")}, '
                             f'the gate scored {ARMS[self.ARM]}')
        self.net = net


def make(arm, rng=None):
    if arm not in ARMS:
        raise ValueError(f'unknown value arm {arm}')
    return type(f'ValueArm_{arm}', (ValueArmPlanner,), {'ARM': arm})(rng)
