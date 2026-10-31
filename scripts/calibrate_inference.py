"""Reproduce the preregistered synthetic inference checks."""
import argparse
import json
from pathlib import Path
import random
import time

from catan_ai.experiments import source_files
from catan_ai.inference import paired_normal_lcb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--alternative-seeds', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1010000)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.alternative_seeds < 128:
        raise ValueError('valid alpha and at least 128 alternative seeds required')
    rng = random.Random(args.seed)
    started = time.monotonic()
    settings = [
        ('symmetric_null', 128, [-.5, .5], [.5, .5]),
        ('sparse_null', 128, [-.25, 0, .25], [.1, .8, .1]),
        ('rare_negative_null', 128, [-.99, .01], [.01, .99]),
        ('rare_positive_null', 128, [-.01, .99], [.99, .01]),
        ('five_point_alternative', args.alternative_seeds, [-.5, .5], [.45, .55]),
    ]
    rows = []
    for name, n, values, weights in settings:
        rejected = sum(paired_normal_lcb(rng.choices(values, weights, k=n), args.alpha) > 0 for _ in range(1000))
        rate = rejected / 1000
        rows.append({'name': name, 'seeds': n, 'values': values, 'weights': weights,
                     'replicates': 1000, 'rejection_rate': rate,
                     'pass': rate >= .8 if 'alternative' in name else rate <= args.alpha + .025})
    result = {'source': source_files(), 'seed': args.seed, 'alpha': args.alpha,
              'seconds': time.monotonic() - started, 'rows': rows,
              'pass': all(row['pass'] for row in rows)}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
