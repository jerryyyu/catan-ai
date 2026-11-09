"""planner_pv2_tiles_ens5 (src/catan_ai/tiles_ensemble.py): five-seed mean-probability ensemble."""
import json

import pytest

np = pytest.importorskip('numpy')

from catan_ai import tiles_ensemble as te  # noqa: E402
from catan_ai import tiles_value as tv  # noqa: E402

have_all = all((te.CHECKPOINTS / f'{s}.npz').exists() for s, _ in te.ENS5)
needs_weights = pytest.mark.skipif(not have_all, reason='ensemble weights are local artifacts (not committed)')


def test_members_are_pinned_in_gate_order():
    assert [s for s, _ in te.ENS5] == ['value_tiles_v1'] + [f'value_tiles_ens_s{k}' for k in (1, 2, 3, 4)]
    served = json.loads((te.CHECKPOINTS / 'value_tiles_v1.meta.json').read_text())
    assert te.ENS5[0][1] == served['pt_sha256']
    for stem, pt in te.ENS5[1:]:
        meta = json.loads((te.CHECKPOINTS / f'{stem}.meta.json').read_text())
        assert meta['pt_sha256'] == pt and meta['arm'] == 'tiles'


def test_logit_of_mean_probability(monkeypatch):
    class Fake:
        def __init__(self, stem):
            self.k = {'a': -1.0, 'b': 2.0}[Path(stem).name]
            self.meta = {'pt_sha256': Path(stem).name, 'npz_sha256': Path(stem).name * 2}

        def logits(self, rows):
            return [self.k] * len(rows)
    from pathlib import Path
    monkeypatch.setattr(tv, 'TilesValueNet', Fake)
    net = te.TilesEnsembleNet(members=(('a', 'a'), ('b', 'b')), root=Path('/x'))
    p = (1 / (1 + np.exp(1.0)) + 1 / (1 + np.exp(-2.0))) / 2
    assert np.allclose(net.logits([[0], [1]]), np.log(p / (1 - p)))
    with pytest.raises(ValueError):
        te.TilesEnsembleNet(members=(('a', 'wrong'),), root=Path('/x'))


@needs_weights
def test_real_ensemble_matches_its_members():
    net = te.TilesEnsembleNet()
    rows = np.random.default_rng(3).normal(size=(7, len(net.nets[0].mean)))
    each = [1 / (1 + np.exp(-np.asarray(m.logits(rows)))) for m in net.nets]
    p = np.mean(each, axis=0)
    assert np.allclose(net.logits(rows), np.log(p / (1 - p)))
    assert len({m.meta['npz_sha256'] for m in net.nets}) == 5
