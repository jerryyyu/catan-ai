import runpy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/calibrate_joint_inference.py'


def test_joint_claim_requires_every_panel_but_coverage_counts_any_false_bound():
    module = runpy.run_path(str(SCRIPT))
    calls = []

    def bounds(samples, alpha):
        calls.append((len(samples), alpha))
        return -0.1 if len(calls) % 4 == 0 else 0.1

    module['calibrate'].__globals__['paired_normal_lcb'] = bounds
    rows = module['calibrate'](replicates=2, alternative_seeds=128)
    assert [r['rate'] for r in rows] == [1, 1, 1, 0]
    assert not any(r['pass'] for r in rows)
    assert all(alpha == .0125 for _, alpha in calls)
    assert len(calls) == 32


def test_binomial_interval_keeps_boundary_uncertainty():
    wilson = runpy.run_path(str(SCRIPT))['wilson']
    assert wilson(0, 1000)[1] > 0
    assert wilson(1000, 1000)[0] < 1
    lo, hi = wilson(500, 1000)
    assert lo < .5 < hi
    with pytest.raises(ValueError):
        wilson(1, 0)
