"""Training-only split-replicate reliability diagnostic; never fits a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from train_paired_value import read_dataset


def summarize(rows):
    if not rows:
        raise ValueError('no training positions')
    samples = []
    groups = {}
    for row in rows:
        base = row['choices'][0]
        weight = 1 / (len(rows) * (len(row['choices']) - 1))
        for choice in row['choices'][1:]:
            differences = [a['win'] - b['win'] for a, b in
                           zip(choice['rollouts'], base['rollouts'], strict=True)]
            if len(differences) != 4:
                raise ValueError('requires four paired replicates')
            left, right = sum(differences[:2]) / 2, sum(differences[2:]) / 2
            mean = sum(differences) / 4
            mean_noise = sum((d - mean) ** 2 for d in differences) / 12
            samples.append((left, right, mean, mean_noise, weight))
            group = groups.setdefault(choice['action']['type'], {'comparisons': 0, 'weight': 0., 'target_sum': 0.})
            group['comparisons'] += 1
            group['weight'] += weight
            group['target_sum'] += weight * mean
    mx = sum(w * x for x, _, _, _, w in samples)
    my = sum(w * y for _, y, _, _, w in samples)
    covariance = sum(w * (x - mx) * (y - my) for x, y, _, _, w in samples)
    vx = sum(w * (x - mx) ** 2 for x, _, _, _, w in samples)
    vy = sum(w * (y - my) ** 2 for _, y, _, _, w in samples)
    correlation = covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None
    for group in groups.values():
        group['mean_advantage'] = group.pop('target_sum') / group['weight']
    return {'positions': len(rows), 'comparisons': len(samples),
            'split_half_covariance': covariance, 'split_half_correlation': correlation,
            'estimated_four_replicate_mean_noise': sum(w * v for _, _, _, v, w in samples),
            'mean_squared_target': sum(w * m * m for _, _, m, _, w in samples),
            'zero_target_fraction': sum(w for _, _, m, _, w in samples if m == 0),
            'by_alternative_type': groups,
            'minimum_split_half_correlation': .1,
            'pass': correlation is not None and correlation >= .1 and covariance > 0,
            'claim': 'training-label reliability diagnostic, not gameplay strength'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('diagnostic already exists')
    rows, producer, hashes = read_dataset(args.root / 'train', 1464000, 64)
    model = json.loads((args.root / 'model.json').read_text())
    if model['train_sha256'] != hashes or model['producer'] != producer:
        raise ValueError('training corpus differs from frozen fit')
    result = {**summarize(rows), 'train_sha256': hashes, 'producer': producer,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('train_sha256', 'producer')}))


if __name__ == '__main__':
    main()
