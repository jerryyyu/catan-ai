"""planner_pv2_tiles_va_p1 (P1 placement) and planner_pv2_tiles_va_t9_explore (T10 data) serving."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
np = pytest.importorskip('numpy')
from catan_ai import value_arms_serving as vas  # noqa: E402

WEIGHTS = all((ROOT / 'checkpoints' / f'value_tiles_va_{a}.npz').exists() for a in ('t9', 'p1'))


def _python_engine():
    from catan_ai.engine import BACKEND
    return BACKEND == 'python'


def test_p1_pin_matches_its_meta():
    m = json.loads((ROOT / 'checkpoints' / 'value_tiles_va_p1.meta.json').read_text())
    assert m['pt_sha256'] == vas.PLACEMENT_ARMS['p1'] and m['value_arm'] == 'p1' and m['parity_max_abs_diff'] < 1e-4


def test_declarations_are_consistent():
    dev = json.loads((ROOT / 'experiments' / 'p1-dev.json').read_text())
    self_ = json.loads((ROOT / 'experiments' / 'p1-self-screen.json').read_text())
    assert self_['candidate'] == 'planner_pv2_tiles_va_p1' and self_['control'] == 'planner_pv2_tiles_va_t9'
    assert self_['panels'][0]['opponents'] == ['planner_pv2_tiles_va_t9'] * 3 and self_['panels'][0]['seed'] == 3400000
    js = {a: json.loads((ROOT / 'experiments' / f'jsettlers-p1-dev-{a}.json').read_text()) for a in ('t9', 'p1')}
    assert js['t9']['shards'] == js['p1']['shards'] == 4 and js['t9']['jsettlers']['seed'] == js['p1']['jsettlers']['seed'] == 20261121
    assert js['t9']['server']['port'] != js['p1']['server']['port']
    t10 = json.loads((ROOT / 'experiments' / 'value-data-t10.json').read_text())
    assert t10['policy'] == 'planner_pv2_tiles_va_t9_explore' and t10['seeds'] == [3500000, 100000]
    assert t10['split'] == {'train': [3500000, 95000], 'test': [3595000, 5000]}
    assert 'self_served' in dev['tables'] and 'jsettlers' in dev['tables']


@pytest.mark.skipif(not WEIGHTS, reason='local weights absent')
def test_p1_places_by_the_net_and_differs_from_the_economic_choice():
    if not _python_engine():
        pytest.skip('the served binding is for the pure-Python engine')
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    differs = 0
    for seed in range(3300001, 3300006):
        game = Game.new(list('abcd'), seed)
        p1, t9 = make_policy('planner_pv2_tiles_va_p1', seed), make_policy('planner_pv2_tiles_va_t9', seed)
        view = Observation.for_policy(game, 0, p1)
        choice = p1.choose(view, 0)
        assert choice in view.legal_actions(0) and choice['type'] == 'build_settlement'
        differs += choice != t9.choose(Observation.for_policy(game, 0, t9), 0)
        assert p1.choose(view, 0) == choice  # deterministic
    assert differs >= 1  # top-1 agreement with the economic choice was .41 on the placement test


@pytest.mark.skipif(not WEIGHTS, reason='local weights absent')
def test_explore_ends_the_turn_at_the_declared_rate_and_replays(monkeypatch):
    if not _python_engine():
        pytest.skip('the served binding is for the pure-Python engine')
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation

    def play(rate, seed=4343):
        monkeypatch.setattr(vas, 'EXPLORE_END_TURN_RATE', rate)
        game = Game.new(list('abcd'), seed)
        bots = [make_policy('planner_pv2_tiles_va_t9_explore', seed * 17 + i + 1) for i in range(4)]
        trace, eligible = [], 0
        while game.winner is None and game.turn < 300:
            s = game.current
            view = Observation.for_policy(game, s, bots[s])
            legal = view.legal_actions(s)
            if view.phase == 'actions' and any(a['type'] == 'end_turn' for a in legal) \
                    and any(a['type'] in vas.ExploreEndTurnPlanner.BUILDS for a in legal):
                eligible += 1
            a = bots[s].choose(view, s)
            trace.append((s, a['type']))
            game.apply(s, a)
        return trace, eligible, sum(b.explored for b in bots)

    t_off, _, ex_off = play(0.0)
    assert ex_off == 0
    t_all, eligible, ex_all = play(1.0)
    assert eligible > 0 and ex_all == eligible  # every eligible decision explored
    assert play(1.0)[0] == t_all and play(0.05)[0] == play(0.05)[0]  # seeded: a replay reproduces the game
