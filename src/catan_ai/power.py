"""Screen power: how small a paired win-rate effect a seed-cluster design can see.

Kept separate from ``inference.py`` so that adding planning helpers does not
invalidate the recorded calibration of the held-out inference implementation.
"""

import math
from statistics import NormalDist, pstdev


def cluster_sigma(values):
    """Population SD of per-cluster paired win differences, each in [-1, 1]."""
    if len(values) < 2 or any(not -1 <= v <= 1 for v in values):
        raise ValueError('at least two bounded cluster differences required')
    return pstdev(values)


def minimum_detectable_effect(sigma, clusters, alpha=.05, power=.8):
    """Smallest true paired lift a one-sided test detects with the given power."""
    if sigma < 0 or clusters < 1 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError('nonnegative sigma, positive clusters and valid alpha/power required')
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return sigma * z / math.sqrt(clusters)


def required_clusters(sigma, effect, alpha=.05, power=.8):
    """Clusters needed so that ``effect`` is the minimum detectable effect."""
    if sigma < 0 or effect <= 0 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError('nonnegative sigma, positive effect and valid alpha/power required')
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return max(2, math.ceil((sigma * z / effect) ** 2))


def panel_differences(rows, panel):
    """Per-seed mean of candidate-minus-baseline wins from evaluator game rows."""
    by_seed = {}
    for row in rows:
        if row['panel'] != panel:
            continue
        candidate = int(row['candidate'].get('winner') == row['seat'])
        baseline = int(row['baseline'].get('winner') == row['seat'])
        by_seed.setdefault(row['seed'], []).append(candidate - baseline)
    return [sum(v) / len(v) for _, v in sorted(by_seed.items())]


def check_design(config):
    """Refuse a screen whose declared power block its seed count cannot deliver.

    ``config['power']`` is optional. When present it names ``target_effect`` and
    ``planning_sigma`` (plus optional ``power``); the per-panel alpha is the
    Bonferroni share used by the evaluator. Returns the required cluster count.
    """
    block = config.get('power')
    if block is None:
        return None
    alpha = .05 / len(config['panels'])
    needed = required_clusters(block['planning_sigma'], block['target_effect'], alpha, block.get('power', .8))
    if config['seeds_per_panel'] < needed:
        raise ValueError(f"underpowered screen: {config['seeds_per_panel']} seeds per panel cannot detect "
                         f"{block['target_effect']} at sigma {block['planning_sigma']}; {needed} required")
    return needed
