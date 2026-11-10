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
