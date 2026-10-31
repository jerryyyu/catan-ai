"""Approximate inference over independent, paired board-seed averages."""

import math
from statistics import NormalDist, mean, stdev


def paired_normal_lcb(values, alpha=.05):
    if len(values) < 128:
        return None
    if any(not -1 <= v <= 1 for v in values) or not 0 < alpha < 1:
        raise ValueError('bounded seed averages and valid alpha required')
    return max(-1, mean(values) - NormalDist().inv_cdf(1 - alpha)
               * max(.25, stdev(values)) / math.sqrt(len(values)))


def planned_seeds(dev_values, effect=.05, alpha=.05, power=.8):
    if len(dev_values) < 2 or effect <= 0:
        raise ValueError('DEV clusters and a positive target effect required')
    # A floor avoids planning a tiny trial from an accidentally constant DEV sample.
    sigma = max(.25, stdev(dev_values))
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return max(128, math.ceil((sigma * z / effect) ** 2))
