import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_arm_is_refused():
    from catan_ai.value_arms_serving import make
    with pytest.raises(ValueError, match='unknown value arm'):
        make('t7')


def test_candidates_are_registered_and_pinned():
    from catan_ai import value_arms_serving as v
    assert set(v.ARMS) == {'t9', 't8', 's1'}
    for arm, pt in v.ARMS.items():
        meta = ROOT / 'checkpoints' / f'value_tiles_va_{arm}.meta.json'
        import json
        m = json.loads(meta.read_text())
        assert m['pt_sha256'] == pt and m['value_arm'] == arm and m['parity_max_abs_diff'] < 1e-4


@pytest.mark.skipif(not (ROOT / 'checkpoints' / 'value_tiles_va_t9.npz').exists(), reason='local weights absent')
@pytest.mark.skipif(__import__('catan_ai.engine', fromlist=['BACKEND']).BACKEND != 'python',
                    reason='the served binding is for the pure-Python engine (Cython runs bind locally)')
def test_candidate_scores_with_its_own_weights():
    pytest.importorskip('numpy')
    from catan_ai.bots import make_policy
    a, b = make_policy('planner_pv2_tiles_va_t9', 1), make_policy('planner_pv2_tiles_buyfix', 1)
    assert a.net.meta['npz_sha256'] != b.net.meta['npz_sha256']
    assert type(a).__mro__[1].__name__ == 'ValueArmPlanner'


def test_exporter_drops_phase_and_refuses_other_sizes(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('numpy')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import arch_bakeoff as ab
    import export_tiles_npz as ex
    net = ab.make_net(torch, 'tiles', phase=True)
    path = tmp_path / 'p.pt'
    torch.save({'arm': 'tiles', 'state': net.state_dict(), 'smoke': False}, path)
    loaded, _ = ex.load_torch_net(torch, path)
    assert hasattr(loaded, 'phase_vec')
    torch.save({'arm': 'tiles', 'state': net.state_dict(), 'smoke': False, 'size': {'d': 256}}, path)
    with pytest.raises(ValueError, match='non-default tiles size'):
        ex.load_torch_net(torch, path)
