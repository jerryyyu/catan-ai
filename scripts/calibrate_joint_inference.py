"""Synthetic familywise coverage and joint-power diagnostic for four panels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from catan_ai.experiments import source_files
from catan_ai.inference import paired_normal_lcb


def wilson(successes, trials):
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError('valid binomial counts required')
    z = 1.959963984540054
    p = successes / trials
    divisor = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / divisor
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / divisor
    return [max(0., center - radius), min(1., center + radius)]


def calibrate(seed=1449001, replicates=1000, alternative_seeds=1600):
    rng = random.Random(seed)
    rows = []
    symmetric = ([-.5, .5], [.5, .5])
    rare_negative = ([-.99, .01], [.01, .99])
    rare_positive = ([-.01, .99], [.99, .01])
    sparse = ([-.25, 0, .25], [.1, .8, .1])
    scenarios = [
        ('independent_null', [symmetric] * 4, 128, False, False),
        ('shared_null', [symmetric] * 4, 128, True, False),
        ('mixed_tail_null', [symmetric, rare_negative, rare_positive, sparse], 128, False, False),
        ('independent_five_point_alternative', [([-.5, .5], [.45, .55])] * 4,
         alternative_seeds, False, True),
    ]
    for name, distributions, n, shared, alternative in scenarios:
        successes = 0
        for _ in range(replicates):
            samples = [rng.choices(values, weights, k=n) for values, weights in distributions[:1 if shared else 4]]
            if shared:
                samples *= 4
            positive = [paired_normal_lcb(sample, .0125) > 0 for sample in samples]
            successes += all(positive) if alternative else any(positive)
        interval = wilson(successes, replicates)
        rows.append({'name': name, 'clusters_per_panel': n, 'distributions': distributions,
                     'replicates': replicates, 'count': successes, 'rate': successes / replicates,
                     'wilson_95': interval,
                     'metric': 'all_four_power' if alternative else 'familywise_null_rejection',
                     'pass': interval[0] >= .8 if alternative else interval[1] <= .075})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    rows = calibrate()
    result = {'source': source_files(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seed': 1449001, 'family_size': 4, 'family_alpha': .05, 'alpha': .0125,
              'seconds': time.monotonic() - started, 'rows': rows,
              'pass': all(row['pass'] for row in rows)}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['rows'], indent=2))


if __name__ == '__main__':
    main()
