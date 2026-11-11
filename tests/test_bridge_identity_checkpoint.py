"""bridge_identity must select the Cython-bound checkpoint before reading it (value-arms held-out, 2026-11-11)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


@pytest.mark.parametrize('module', ['jsettlers_smoke', 'stac_match'])
def test_selection_happens_before_the_checkpoint_is_read(module, monkeypatch):
    import catan_ai.local_checkpoint as lc
    import catan_ai.parameter_planner as pp
    calls = []
    monkeypatch.setattr(lc, 'select_parameter_checkpoint', lambda *a, **k: calls.append('select'))
    def load(*a, **k):
        calls.append('load')
        return [0.0], 'h', Path('p')
    monkeypatch.setattr(pp, 'load_parameters', load)
    mod = __import__(module)
    out = mod.bridge_identity('planner_pv2_tiles')
    assert calls[:2] == ['select', 'load'] and out['checkpoint_model_hash'] == 'h'
