"""Approximate inference over independent, paired board-seed averages."""

import math
from statistics import NormalDist, mean, stdev


DEFAULT_SD_FLOOR = .25
# A floor below the default is allowed only for large samples (owner decision 2026-11-10: "We can relax the
# floor"), where the sample SD is a stable estimate; each config declares its floor and the calibration
# receipt must match it.
MIN_SEEDS_BELOW_DEFAULT_FLOOR = 512


def paired_normal_lcb(values, alpha=.05, sd_floor=DEFAULT_SD_FLOOR):
    if len(values) < 128:
        return None
    if any(not -1 <= v <= 1 for v in values) or not 0 < alpha < 1:
        raise ValueError('bounded seed averages and valid alpha required')
    if not 0 <= sd_floor <= 1:
        raise ValueError('sd_floor must lie in [0, 1]')
    if sd_floor < DEFAULT_SD_FLOOR and len(values) < MIN_SEEDS_BELOW_DEFAULT_FLOOR:
        raise ValueError(f'an SD floor below {DEFAULT_SD_FLOOR} needs at least {MIN_SEEDS_BELOW_DEFAULT_FLOOR} seeds')
    return max(-1, mean(values) - NormalDist().inv_cdf(1 - alpha)
               * max(sd_floor, stdev(values)) / math.sqrt(len(values)))


def planned_seeds(dev_values, effect=.05, alpha=.05, power=.8):
    if len(dev_values) < 2 or effect <= 0:
        raise ValueError('DEV clusters and a positive target effect required')
    # A floor avoids planning a tiny trial from an accidentally constant DEV sample.
    sigma = max(.25, stdev(dev_values))
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return max(128, math.ceil((sigma * z / effect) ** 2))
