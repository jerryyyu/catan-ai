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
    's1': '3096f12ec645ccbda8359c45ad95ffc29e9b71853fddfed7fa0eb5178a5d3101',  # offline FAIL; screened by owner decision
}
# placement-value arms (experiments/placement-value.json): the .pt sha the placement gate scored
PLACEMENT_ARMS = {'p1': '95571cf567db6cc74a13fe790bd7372e748fa3b8e4f76fc8181a8368d4ecc4b0'}
EXPLORE_END_TURN_RATE = 0.05  # experiments/value-data-t10.json


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


def _load_pinned(stem, pins, arm):
    from .tiles_value import TilesValueNet
    net = TilesValueNet(CHECKPOINTS / f'value_tiles_va_{stem}')
    if net.meta.get('pt_sha256') != pins[arm] or net.meta.get('value_arm') != arm:
        raise ValueError(f'value_tiles_va_{stem}: exported from {net.meta.get("pt_sha256")}, the gate scored {pins[arm]}')
    return net


class PlacementValuePlanner(ValueArmPlanner):
    """``planner_pv2_tiles_va_p1``: the T9 candidate (served) with its two setup settlements chosen by the P1 value net
    (experiments/placement-value.json, offline PASS) over the afterstate of each legal vertex, instead of the fitted
    economic score. Everything else, including the actions-phase value net (T9), is the served bot's, so a screen of
    this policy measures placement alone. Setup has no hidden hands, so the chooser's world is exact."""

    ARM = 't9'

    def __init__(self, rng=None):
        super().__init__(rng)
        self.setup_net = _load_pinned('p1', PLACEMENT_ARMS, 'p1')

    def choose(self, view, seat):
        if view.phase != 'setup_settlement':
            return super().choose(view, seat)
        from copy import deepcopy
        from .encoding import encode
        from .observation import Observation
        legal = view.legal_actions(seat)
        world = self._world(view)
        rows = []
        for action in legal:
            child = deepcopy(world, {id(world.board): world.board})
            child.apply(seat, action)
            rows.append(encode(Observation.from_game(child, seat, include_events=False), child))
        logits = [float(v) for v in self.setup_net.logits(rows)]
        return legal[max(range(len(legal)), key=logits.__getitem__)]


class ExploreEndTurnPlanner(ValueArmPlanner):
    """``planner_pv2_tiles_va_t9_explore``: the served T9 for data generation (experiments/value-data-t10.json) that,
    with probability ``EXPLORE_END_TURN_RATE`` at each actions-phase decision where ending the turn and a city or
    settlement build are both legal, ends the turn through the served end-of-turn path instead. The draw uses the
    policy's own seeded rng, so a replay from the game seed reproduces it. It gives the value data the "end now with a
    buildable hand" rows the recipe review found missing (board #1 comment 6771511862)."""

    ARM = 't9'
    BUILDS = ('build_city', 'build_settlement')

    def __init__(self, rng=None):
        super().__init__(rng)
        self.explored = 0

    def choose(self, view, seat):
        if view.phase == 'actions':
            legal = view.legal_actions(seat)
            if (any(a['type'] == 'end_turn' for a in legal) and any(a['type'] in self.BUILDS for a in legal)
                    and self.rng.random() < EXPLORE_END_TURN_RATE):
                self.explored += 1
                action = (self._fallback(view, seat) if any(a['type'] == 'play_development' for a in legal)
                          else {'type': 'end_turn'})
                return self._end_of_turn_offer(view, seat, action)
        return super().choose(view, seat)


def make(arm, rng=None):
    if arm == 'p1':
        return PlacementValuePlanner(rng)
    if arm == 't9_explore':
        return ExploreEndTurnPlanner(rng)
    if arm not in ARMS:
        raise ValueError(f'unknown value arm {arm}')
    return type(f'ValueArm_{arm}', (ValueArmPlanner,), {'ARM': arm})(rng)
