import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import arch_bakeoff as ab  # noqa: E402
import tiles_onpolicy_gate as g  # noqa: E402


def test_declaration_resolves_split_and_data_keys():
    cfg = json.loads(g.DECL.read_text())
    assert ab.declared_split(cfg) == {'train': [3100000, 95000], 'test': [3195000, 5000]}
    ref = json.loads((ROOT / 'experiments' / 'value-v2-tiles.json').read_text())
    assert ref['split'] == ab.declared_split(cfg) and ref['roots'][0]['root'] == 'value-tiles-v1'
    base = json.loads((ROOT / 'experiments' / 'arch-bakeoff.json').read_text())
    assert cfg['common'] == base['common'] and cfg['arms']['tiles'] == base['arms']['tiles']


def test_paired_auc_is_zero_for_identical_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    p = rng.random(400)
    out = g.paired_auc(p, p, y, np.repeat(np.arange(100), 4))
    assert out['diff'] == 0 and out['ci95'] == [0, 0]


@pytest.mark.parametrize('mutate, expect', [
    (lambda r: r.update(models_sha256=['other']), 'models'),
    (lambda r: r.update(tta=True), 'symmetry'),
    (lambda r: r.update(script_sha256='old'), 'script'),
    (lambda r: r.update(test_corpus_sha256='X'), 'corpus'),
])
def test_xeval_binding(mutate, expect):
    r = {'models_sha256': ['m'], 'tta': False, 'script_sha256': 'S', 'test_corpus_sha256': 'C'}
    assert g.bind_xeval(dict(r), 'm', 'S', 'C') == []
    bad = dict(r)
    mutate(bad)
    assert any(expect in p for p in g.bind_xeval(bad, 'm', 'S', 'C'))
    assert any('corpus' in p for p in g.bind_xeval(dict(r), 'm', 'S', None))
    assert g.bind_xeval(None, 'm', 'S', 'C') == ['receipt missing']
