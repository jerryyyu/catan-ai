"""Declared variance floor for paired-normal gates (owner 2026-11-10: "We can relax the floor")."""
import random
import statistics

import pytest

from catan_ai.inference import paired_normal_lcb


def test_default_floor_reproduces_the_previous_rule():
    rng = random.Random(3)
    values = [rng.choice([-.25, 0, .25]) for _ in range(600)]
    sd = statistics.stdev(values)
    assert sd < .25
    assert paired_normal_lcb(values, .05) == paired_normal_lcb(values, .05, .25)


def test_lower_floor_uses_the_sample_sd_and_needs_512_seeds():
    rng = random.Random(4)
    values = [rng.choice([-.25, 0, .25]) + .01 for _ in range(600)]
    assert paired_normal_lcb(values, .05, .1) > paired_normal_lcb(values, .05, .25)  # sample SD < .25 now counts
    with pytest.raises(ValueError, match='512'):
        paired_normal_lcb(values[:300], .05, .1)
    with pytest.raises(ValueError):
        paired_normal_lcb(values, .05, -.1)


def _design(floor=None, seeds=512, sigma=.02, alternative=.005):
    config = {'inference': 'paired_normal', 'seeds_per_panel': seeds, 'panels': [{}],
              'gates': {'minimum_paired_win_lift': 0, 'minimum_paired_lcb': 0},
              'power': {'alternative_effect': alternative, 'planning_sigma': sigma}}
    if floor is not None:
        config['variance_floor'] = floor
    return config


def test_admission_margin_uses_the_declared_floor_not_a_smaller_planning_sigma():
    """Codex HOLD @301c8e3: 512 seeds, floor .10, sigma .02, alternative .005 is underpowered."""
    from catan_ai.power import check_design
    with pytest.raises(ValueError, match='underpowered'):
        check_design(_design(floor=.1))
    record = check_design(_design(floor=.1, alternative=.02))
    assert record['variance_floor'] == .1
    assert record['pass_threshold'] == pytest.approx(1.6448536 * .1 / 512 ** .5, rel=1e-6)


def test_admission_refuses_a_lower_floor_below_512_seeds():
    from catan_ai.power import check_design
    with pytest.raises(ValueError, match='512'):
        check_design(_design(floor=.1, seeds=400, alternative=.05))
    assert check_design(_design(seeds=400, sigma=.3, alternative=.08))['variance_floor'] == .25


def test_heldout_calibration_must_match_the_declared_floor():
    from catan_ai.search_eval import check_calibration
    source = {'inference.py': 'h'}
    calibration = {'pass': True, 'source': {'inference.py': 'h'}, 'alpha': .05, 'sd_floor': .25}
    config = {'panels': [{}], 'variance_floor': .1}
    with pytest.raises(ValueError, match='variance_floor'):
        check_calibration(config, calibration, source)
    check_calibration(config, dict(calibration, sd_floor=.1), source)
    check_calibration({'panels': [{}]}, calibration, source)
