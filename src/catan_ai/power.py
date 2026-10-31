"""Screen power: can a seed-cluster design pass its own configured gate?

Two different questions live here and must not be confused:

- ``check_design`` (admission). Under a declared *alternative* true lift, what
  is the probability that every panel clears the configured decision rule (the
  point-estimate bar ``minimum_paired_win_lift`` and, when configured, the
  ``minimum_paired_lcb`` gate under the configured inference)? The evaluator
  refuses a design whose family pass probability is below the declared power.
- ``minimum_detectable_effect`` (description). The smallest lift a one-sided
  test against *zero* detects with 80% power. It is a sensitivity diagnostic
  only; it says nothing about clearing a +5 pp bar.

Both use a normal approximation to the mean of independent per-seed cluster
differences. Kept separate from ``inference.py`` so planning helpers do not
invalidate the recorded calibration of the held-out inference implementation.
"""

import math
from statistics import NormalDist, pstdev

_N = NormalDist()
_MAX_CLUSTERS = 1_000_000


def _finite(name, value, low=-math.inf, high=math.inf, low_open=False, high_open=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < low or value > high or (low_open and value == low) or (high_open and value == high):
        raise ValueError(f'{name} out of range')
    return float(value)


def cluster_sigma(values):
    """Population SD of per-cluster paired win differences, each in [-1, 1]."""
    if len(values) < 2 or any(not -1 <= v <= 1 for v in values):
        raise ValueError('at least two bounded cluster differences required')
    return pstdev(values)


def minimum_detectable_effect(sigma, clusters, alpha=.05, power=.8):
    """Zero-null sensitivity: smallest true lift a one-sided test vs 0 detects."""
    sigma = _finite('sigma', sigma, 0)
    _finite('alpha', alpha, 0, 1, True, True)
    _finite('power', power, 0, 1, True, True)
    if not isinstance(clusters, int) or clusters < 1:
        raise ValueError('positive integer cluster count required')
    z = _N.inv_cdf(1 - alpha) + _N.inv_cdf(power)
    return sigma * z / math.sqrt(clusters)


def required_clusters(sigma, effect, alpha=.05, power=.8):
    """Zero-null sensitivity: clusters for which ``effect`` is the detectable effect."""
    sigma = _finite('sigma', sigma, 0)
    effect = _finite('effect', effect, 0, 2, True)
    _finite('alpha', alpha, 0, 1, True, True)
    _finite('power', power, 0, 1, True, True)
    z = _N.inv_cdf(1 - alpha) + _N.inv_cdf(power)
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


def gate_threshold(gates, inference, panels, clusters, sigma):
    """The point-estimate value a panel's lift must reach to pass every lift gate.

    The point bar applies directly. An LCB gate ``lcb > L`` becomes ``lift > L +
    margin``: Hoeffding's margin is fixed by the cluster count, and the paired
    normal margin is approximated with the planning sigma.
    """
    threshold = _finite('minimum_paired_win_lift', gates['minimum_paired_win_lift'], -1, 1)
    if 'minimum_paired_lcb' in gates:
        floor = _finite('minimum_paired_lcb', gates['minimum_paired_lcb'], -1, 1)
        alpha = .05 / panels
        if inference == 'hoeffding':
            margin = math.sqrt(2 * math.log(panels / .05) / clusters)
        elif inference == 'paired_normal':
            margin = _N.inv_cdf(1 - alpha) * sigma / math.sqrt(clusters)
        else:
            raise ValueError('unknown inference method')
        threshold = max(threshold, floor + margin)
    elif inference not in ('hoeffding', 'paired_normal'):
        raise ValueError('unknown inference method')
    return threshold


def family_pass_probability(alternative, sigma, clusters, gates, inference, panels):
    """Probability that all ``panels`` independent panels pass at true lift ``alternative``."""
    threshold = gate_threshold(gates, inference, panels, clusters, sigma)
    if sigma == 0:
        panel = 1.0 if alternative >= threshold else 0.0
    else:
        panel = _N.cdf((alternative - threshold) * math.sqrt(clusters) / sigma)
    return panel ** panels, panel, threshold


def check_design(config):
    """Refuse a screen that is unlikely to pass its own gate at the declared alternative.

    ``config['power']`` is optional. When present it declares
    ``alternative_effect`` (the true paired lift the screen is meant to find),
    ``planning_sigma`` (cluster SD) and optional ``power`` (family pass
    probability, default .8). Returns the admission record, or ``None`` when no
    block is declared. A legacy ``target_effect`` block planned against zero and
    is refused rather than reinterpreted.
    """
    block = config.get('power')
    if block is None:
        return None
    if 'target_effect' in block and 'alternative_effect' not in block:
        raise ValueError('legacy power block (target_effect) planned a zero-null test; declare '
                         'alternative_effect above the configured gate instead')
    alternative = _finite('alternative_effect', block.get('alternative_effect'), -1, 1, True)
    sigma = _finite('planning_sigma', block.get('planning_sigma'), 0, 1, True)
    target = _finite('power', block.get('power', .8), 0, 1, True, True)
    clusters = config.get('seeds_per_panel')
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 2:
        raise ValueError('seeds_per_panel must be an integer of at least 2')
    panels = len(config['panels'])
    if panels < 1:
        raise ValueError('at least one panel required')
    gates = config['gates']
    inference = config.get('inference', 'hoeffding')
    achieved, per_panel, threshold = family_pass_probability(alternative, sigma, clusters, gates, inference, panels)
    needed = None
    for n in _doubling_search(lambda n: family_pass_probability(alternative, sigma, n, gates, inference, panels)[0] >= target):
        needed = n
    record = {'alternative_effect': alternative, 'planning_sigma': sigma, 'power': target,
              'inference': inference, 'panels': panels, 'clusters': clusters,
              'pass_threshold': threshold, 'panel_pass_probability': per_panel,
              'family_pass_probability': achieved, 'required_clusters': needed}
    if achieved < target:
        raise ValueError(f'underpowered screen: {clusters} seeds per panel pass every panel with probability '
                         f'{achieved:.3f} at true lift {alternative} (threshold {threshold:.4f}, sigma {sigma}); '
                         f'{needed if needed else "no feasible count"} required for {target}')
    return record


def _doubling_search(ok):
    """Yield the smallest n in [2, _MAX_CLUSTERS] with ok(n), if any."""
    low, high = 1, 2
    while high <= _MAX_CLUSTERS and not ok(high):
        low, high = high, high * 2
    if high > _MAX_CLUSTERS:
        if not ok(_MAX_CLUSTERS):
            return
        high = _MAX_CLUSTERS
    while high - low > 1:
        middle = (low + high) // 2
        if ok(middle):
            high = middle
        else:
            low = middle
    yield high
