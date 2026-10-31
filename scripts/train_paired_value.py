"""Fixed ridge fit and separate validation of paired continuation action values."""
import argparse
import hashlib
import json
import math
from pathlib import Path


RECIPE = {'ridge': .1, 'margin': .05, 'feature_dimension': 143,
          'minimum_validation_positions': 48, 'minimum_mse_reduction': .05,
          'minimum_selected_advantage': .02, 'minimum_override_fraction': .05}


def producer(plan):
    return {key: plan[key] for key in ('source', 'script_sha256', 'states_per_board',
                                      'replicates', 'max_actions', 'process_cap')}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_dataset(directory, seed_start, board_count):
    plan = json.loads((directory / 'plan.json').read_text())
    if (plan['seed'] != seed_start or plan['boards'] != board_count
            or plan['states_per_board'] != 4 or plan['replicates'] != 4
            or plan['max_actions'] != 4):
        raise ValueError('dataset differs from the predeclared collection')
    rows, hashes = [], {}
    for seed in range(seed_start, seed_start + board_count):
        path = directory / f'{seed}.json'
        raw = path.read_bytes()
        board = json.loads(raw)
        hashes[str(seed)] = hashlib.sha256(raw).hexdigest()
        if board['identity'] != plan or board['board_seed'] != seed or board['label_truncations'] != 0:
            raise ValueError('board identity or completion mismatch')
        if not 1 <= len(board['rows']) <= 4:
            raise ValueError('board has missing or excess positions')
        for index, row in enumerate(board['rows']):
            if (row['board_seed'] != seed or row['seat'] != index % 4
                    or row['turn'] < 8 + 20 * index
                    or row['replicate_seeds'] != [seed * 10000 + index * 4 + r for r in range(4)]):
                raise ValueError('position or paired seed mismatch')
            if not finite(row['seconds']) or row['seconds'] < 0:
                raise ValueError('invalid collection timing')
            choices = row['choices']
            if not 2 <= len(choices) <= 4 or len({json.dumps(c['action'], sort_keys=True) for c in choices}) != len(choices):
                raise ValueError('missing or duplicate candidate actions')
            for choice in choices:
                if len(choice['features']) != 143 or not all(finite(x) for x in choice['features']):
                    raise ValueError('invalid feature vector')
                rollouts = choice['rollouts']
                if len(rollouts) != 4:
                    raise ValueError('missing continuation labels')
                for result in rollouts:
                    if (result['truncated'] is not False or type(result['winner']) is not int
                            or result['winner'] not in range(4) or type(result['win']) is not int
                            or result['win'] != int(result['winner'] == row['seat'])
                            or type(result['turns']) is not int or result['turns'] < 0):
                        raise ValueError('invalid or incomplete continuation outcome')
                mean = sum(r['win'] for r in rollouts) / 4
                if not finite(choice['mean_win']) or choice['mean_win'] != mean:
                    raise ValueError('stored win target disagrees with raw outcomes')
            rows.append(row)
    return rows, producer(plan), hashes


def pairs(rows):
    for row in rows:
        base = row['choices'][0]
        weight = 1 / (len(rows) * (len(row['choices']) - 1))
        for choice in row['choices'][1:]:
            delta = [x - y for x, y in zip(choice['features'], base['features'], strict=True)]
            yield delta, choice['mean_win'] - base['mean_win'], weight


def cholesky_solve(matrix, target):
    n = len(target)
    lower = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0 or not math.isfinite(value):
                    raise ValueError('ridge system is not positive definite')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    intermediate = [0.] * n
    for i in range(n):
        intermediate[i] = (target[i] - sum(lower[i][j] * intermediate[j] for j in range(i))) / lower[i][i]
    weights = [0.] * n
    for i in reversed(range(n)):
        weights[i] = (intermediate[i] - sum(lower[j][i] * weights[j] for j in range(i + 1, n))) / lower[i][i]
    return weights


def fit(rows):
    if not rows:
        raise ValueError('training data is empty')
    examples = list(pairs(rows))
    dim = len(examples[0][0])
    scales = [math.sqrt(sum(w * x[i] ** 2 for x, _, w in examples)) or 1. for i in range(dim)]
    matrix = [[RECIPE['ridge'] if i == j else 0. for j in range(dim)] for i in range(dim)]
    target = [0.] * dim
    for x, y, weight in examples:
        z = [v / scale for v, scale in zip(x, scales, strict=True)]
        for i in range(dim):
            target[i] += weight * z[i] * y
            for j in range(i + 1):
                matrix[i][j] += weight * z[i] * z[j]
    for i in range(dim):
        for j in range(i):
            matrix[j][i] = matrix[i][j]
    return {'weights': cholesky_solve(matrix, target), 'scales': scales}


def score(model, features):
    return sum(w * x / s for w, x, s in zip(model['weights'], features, model['scales'], strict=True))


def evaluate(model, rows):
    if not rows:
        raise ValueError('validation data is empty')
    error = baseline_error = 0.
    for x, target, weight in pairs(rows):
        error += weight * (score(model, x) - target) ** 2
        baseline_error += weight * target ** 2
    decisions = []
    for row in rows:
        choices = row['choices']
        values = [score(model, c['features']) for c in choices]
        best = max(range(len(choices)), key=lambda i: values[i])
        chosen = best if values[best] - values[0] >= RECIPE['margin'] else 0
        decisions.append({'board_seed': row['board_seed'], 'step': row['step'],
                          'choice': chosen, 'predicted_advantage': values[chosen] - values[0],
                          'observed_advantage': choices[chosen]['mean_win'] - choices[0]['mean_win']})
    reduction = 1 - error / baseline_error if baseline_error > 0 else None
    advantage = sum(d['observed_advantage'] for d in decisions) / len(decisions)
    fraction = sum(d['choice'] != 0 for d in decisions) / len(decisions)
    passed = (len(rows) >= RECIPE['minimum_validation_positions'] and reduction is not None
              and reduction >= RECIPE['minimum_mse_reduction']
              and advantage >= RECIPE['minimum_selected_advantage']
              and fraction >= RECIPE['minimum_override_fraction'])
    return {'positions': len(rows), 'mse': error, 'zero_advantage_mse': baseline_error,
            'mse_reduction': reduction, 'selected_advantage': advantage,
            'override_fraction': fraction, 'pass': passed, 'decisions': decisions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('fit', 'validate'), required=True)
    parser.add_argument('--checkpoint-sha')
    args = parser.parse_args()
    model_path = args.root / 'model.json'
    trainer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.mode == 'fit':
        if model_path.exists():
            raise FileExistsError('checkpoint already frozen')
        rows, identity, hashes = read_dataset(args.root / 'train', 1464000, 64)
        model = {'schema': 1, **fit(rows), 'recipe': RECIPE, 'producer': identity,
                 'train_sha256': hashes, 'training_positions': len(rows), 'trainer_sha256': trainer_hash}
        with model_path.open('x') as stream:
            json.dump(model, stream, indent=2, allow_nan=False)
        print(json.dumps({'positions': len(rows), 'checkpoint_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}))
    else:
        raw = model_path.read_bytes()
        if not args.checkpoint_sha or hashlib.sha256(raw).hexdigest() != args.checkpoint_sha:
            raise ValueError('validation requires the frozen checkpoint hash')
        model = json.loads(raw)
        if model['recipe'] != RECIPE or model['trainer_sha256'] != trainer_hash:
            raise ValueError('fit recipe or trainer changed after freezing')
        if (len(model['weights']) != 143 or len(model['scales']) != 143
                or not all(finite(w) for w in model['weights'])
                or not all(finite(s) and s > 0 for s in model['scales'])):
            raise ValueError('invalid model vectors')
        output = args.root / 'validation.json'
        if output.exists():
            raise FileExistsError('validation readout already exists')
        rows, identity, hashes = read_dataset(args.root / 'validation', 1465000, 16)
        if identity != model['producer']:
            raise ValueError('validation producer differs from training')
        report = {**evaluate(model, rows), 'checkpoint_sha256': args.checkpoint_sha,
                  'validation_sha256': hashes, 'recipe': RECIPE, 'trainer_sha256': trainer_hash}
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({k: v for k, v in report.items() if k not in ('decisions', 'validation_sha256')}))


if __name__ == '__main__':
    main()
