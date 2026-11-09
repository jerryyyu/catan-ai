import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import arch_bakeoff as ab  # noqa: E402
import tiles_tta_ens as te  # noqa: E402
from test_arch_bakeoff import make_fixture  # noqa: E402
from train_value_v2 import symmetry_index  # noqa: E402


def test_seed_config_changes_only_the_seed():
    base = json.loads(te.BASE.read_text())
    for k in (1, 2, 3, 4):
        cfg = te.seed_config(k)
        assert cfg['common']['seed'] == k
        cfg['common']['seed'] = base['common']['seed']
        assert cfg == base


def test_twelve_symmetries_are_permutations():
    perms = symmetry_index(torch).numpy()
    assert perms.shape[0] == 12
    for p in perms:
        assert sorted(p.tolist()) == list(range(ab.WIDTH))


def test_predict_averages_models_and_symmetries():
    x = np.random.default_rng(0).normal(size=(6, ab.WIDTH)).astype(np.float16)
    rows = np.arange(6)
    perms = symmetry_index(torch).numpy()

    class Net(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, b):
            return (b[:, 0] * self.k,)  # log-sigmoid head shape as arch_bakeoff's tiles arm expects
    import types
    saved = {'mean': torch.zeros(ab.WIDTH), 'sd': torch.ones(ab.WIDTH)}
    calls = []
    real_score = ab.score

    def fake_score(torch_, net, arm, xx, rr, mean, sd, device):
        calls.append(xx[0, 0])
        return np.full(len(rr), float(net.k)), None
    ab.score = fake_score
    try:
        out = te.predict(torch, [(types.SimpleNamespace(k=0.2), saved), (types.SimpleNamespace(k=0.6), saved)],
                         x, rows, 'cpu', True, perms)
    finally:
        ab.score = real_score
    assert np.allclose(out, 0.4) and len(calls) == 24  # 2 models x 12 symmetries


def test_smoke_train_seed_and_gate(tmp_path):
    data, _ = make_fixture(tmp_path)
    ref_dir = tmp_path / 'ref'
    ref_dir.mkdir()
    from types import SimpleNamespace
    ab.train(SimpleNamespace(arm='tiles', data=data, records=None, out=ref_dir, device='cpu', max_rows=200, epochs=1),
             json.loads(te.BASE.read_text()))
    seeds = []
    for k in (1, 2, 3, 4):
        out = tmp_path / f'seed{k}'
        te.main(['train-seed', '--seed', str(k), '--data', str(data), '--out', str(out), '--device', 'cpu',
                 '--max-rows', '200', '--epochs', '1'])
        seeds.append(out / 'tiles.pt')
    (tmp_path / 'x').mkdir()
    te.main(['gate', '--data', str(data), '--reference', str(ref_dir / 'tiles.pt'), '--seeds', *map(str, seeds),
             '--xeval-dir', str(tmp_path / 'x'), '--out', str(tmp_path / 'gate.json'), '--device', 'cpu', '--max-rows', '200'])
    g = json.loads((tmp_path / 'gate.json').read_text())
    assert set(g['arms']) == {'reference', 'tta', 'ens5', 'ens5_tta'} and g['smoke']
    assert all(g['arms'][a]['pass'] is None for a in te.ARMS)  # no cross-evals: never a pass
    with pytest.raises(ValueError):
        te.main(['train-seed', '--seed', '0', '--data', str(data), '--out', str(tmp_path / 's0'), '--device', 'cpu'])
