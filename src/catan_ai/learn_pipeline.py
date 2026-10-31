"""Bounded game shards, search targets, training and separate prediction evaluation."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import random
import time

from .bots import make_policy
from .engine import Game
from .experiments import source_files
from .learning import SCHEMA, KINDS, CARDS, action_features, state_features, dot, sigmoid, probabilities, load_model
from .observation import Observation

SPLITS = {'train': (1230000, 1230099), 'validation': (1231000, 1231029), 'test': (1232000, 1232029)}


def producer_source():
    dependencies = {'engine.py', '_engine_py.py', 'engine_backend', 'engine_binary', 'scoring_binary',
                    'scoring.py', 'bots.py', 'observation.py', 'tactical.py',
                    'strategic.py', 'styles.py', 'search.py', 'belief.py'}
    result = {name: sha for name, sha in source_files().items() if name in dependencies}
    for function in (generate, save, action_features, state_features, producer_source):
        result[function.__name__] = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    result['splits'] = hashlib.sha256(json.dumps(SPLITS, sort_keys=True).encode()).hexdigest()
    result['features'] = hashlib.sha256(json.dumps([SCHEMA, KINDS, CARDS]).encode()).hexdigest()
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def generate(root, split, count, targets):
    low, high = SPLITS[split]
    if not 1 <= count <= high - low + 1 or not 1 <= targets <= 16:
        raise ValueError('bounded count/target limits exceeded')
    source = producer_source()
    for seed in range(low, low + count):
        path = root / split / f'{seed}.json'
        identity = {'seed': seed, 'split': split, 'targets': targets, 'source': source,
                    'teacher': 'search_fast', 'policies': ['tactical', 'builder', 'expander', 'heuristic']}
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['identity'] != identity or not existing['complete']:
                raise ValueError('incompatible existing shard')
            continue
        started = time.perf_counter()
        names = identity['policies']
        # Rotate the style assigned to each seat across entire games.
        names = names[seed % 4:] + names[:seed % 4]
        game = Game.new(names, seed)
        bots = [make_policy(n, seed * 17 + s) for s, n in enumerate(names)]
        teachers = [make_policy('search_fast', seed * 31 + s) for s in range(4)]
        rows, attempted, trace = [], 0, hashlib.sha256()
        trajectory, teacher_diagnostics = [], []
        for step in range(10000):
            if game.winner is not None or game.turn >= 2000:
                break
            seat = game.current
            view = Observation.for_policy(game, seat, bots[seat])
            if (game.phase == 'actions' and len(view.actions) > 1 and game.turn >= 8
                    and step % 7 == 0 and attempted < targets):
                observed = Observation.from_game(game, seat)
                teachers[seat].choose(observed, seat)
                diagnostic = teachers[seat].diagnostics[-1]
                teacher_diagnostics.append({'seat': seat, 'step': step, 'diagnostic': diagnostic})
                attempted += 1
                values = diagnostic['values']
                # Failed/incomplete searches never become fabricated teacher labels.
                if (diagnostic['truncated'] == 0 and len(values) >= 2 and
                        all(v['completed'] == v['samples'] and v['completed'] > 0
                            and v['action'] in observed.actions for v in values)):
                    rows.append({'seat': seat, 'turn': game.turn, 'step': step,
                                 'state': state_features(observed),
                                 'actions': [v['action'] for v in values],
                                 'features': [action_features(observed, v['action']) for v in values],
                                 'teacher': values, 'diagnostic': diagnostic})
            action = bots[seat].choose(view, seat)
            trajectory.append({'seat': seat, 'action': action})
            trace.update(json.dumps([seat, action], sort_keys=True).encode())
            game.apply(seat, action)
        for row in rows:
            row['win'] = int(game.winner == row['seat']) if game.winner is not None else None
        if producer_source() != source:
            raise RuntimeError('source changed during generation')
        save(path, {'identity': identity, 'host': platform.node(), 'process_cap': 1,
                    'seconds': time.perf_counter() - started, 'complete': game.winner is not None,
                    'winner': game.winner, 'turns': game.turn, 'attempted': attempted,
                    'trace_sha256': trace.hexdigest(), 'trajectory': trajectory,
                    'teacher_diagnostics': teacher_diagnostics, 'rows': rows})
        print(json.dumps({'seed': seed, 'rows': len(rows), 'complete': game.winner is not None}), flush=True)


def read_split(root, split):
    rows, files = [], {}
    low, high = SPLITS[split]
    source = None
    for path in sorted((root / split).glob('*.json')):
        shard = json.loads(path.read_text())
        identity = shard['identity']
        if not shard['complete'] or identity['split'] != split or not low <= identity['seed'] <= high:
            raise ValueError('invalid or cross-split shard')
        if path.stem != str(identity['seed']):
            raise ValueError('seed identity mismatch')
        if source is not None and source != identity['source']:
            raise ValueError('mixed dataset producers')
        source = identity['source']
        files[str(path)] = digest(path)
        rows.extend(shard['rows'])
    if not rows:
        raise ValueError('no usable examples')
    return rows, files


def targets(row):
    return probabilities([(v['wins'] + .5) / (v['completed'] + 1) / .25 for v in row['teacher']])


def train(root, output, epochs, seed):
    rows, files = read_split(root, 'train')
    if not 1 <= epochs <= 100:
        raise ValueError('training budget exceeds 100 epochs')
    policy, value = [0.] * 26, [0.] * 21
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(epochs):
        rng.shuffle(rows)
        for row in rows:
            predicted = probabilities([dot(policy, x) for x in row['features']])
            desired = targets(row)
            for j in range(len(policy)):
                gradient = sum((p - t) * x[j] for p, t, x in zip(predicted, desired, row['features']))
                policy[j] -= .05 * (gradient + .001 * policy[j])
            error = sigmoid(dot(value, row['state'])) - row['win']
            for j in range(len(value)):
                value[j] -= .05 * (error * row['state'][j] + .001 * value[j])
    save(output, {'schema': SCHEMA, 'policy_weights': policy, 'value_weights': value,
                  'config': {'epochs': epochs, 'seed': seed, 'learning_rate': .05, 'l2': .001,
                             'teacher_temperature': .25}, 'training_files': files,
                  'data_source': json.loads(Path(next(iter(files))).read_text())['identity']['source'],
                  'source': source_files(), 'host': platform.node(), 'process_cap': 1,
                  'seconds': time.perf_counter() - started, 'examples': len(rows)})


def evaluate(root, split, model_path, output):
    if split == 'train':
        raise ValueError('prediction evaluation requires a held-out game split')
    rows, files = read_split(root, split)
    model, model_hash = load_model(model_path)
    if any(json.loads(Path(path).read_text())['identity']['source'] != model['data_source'] for path in files):
        raise ValueError('evaluation producer differs from training producer')
    probabilities_value, losses, uniform_losses, agreement = [], [], [], []
    for row in rows:
        p = probabilities([dot(model['policy_weights'], x) for x in row['features']])
        t = targets(row)
        losses.append(-sum(target * math.log(max(pred, 1e-12)) for target, pred in zip(t, p)))
        uniform_losses.append(math.log(len(p)))
        agreement.append(int(max(range(len(p)), key=p.__getitem__) == max(range(len(t)), key=t.__getitem__)))
        probabilities_value.append(sigmoid(dot(model['value_weights'], row['state'])))
    n = len(rows)
    bins = []
    for i in range(10):
        members = [(p, row['win']) for p, row in zip(probabilities_value, rows) if min(9, int(p * 10)) == i]
        if members:
            bins.append({'bin': i, 'count': len(members),
                         'predicted': sum(p for p, _ in members) / len(members),
                         'observed': sum(y for _, y in members) / len(members)})
    save(output, {'split': split, 'model_hash': model_hash, 'data_files': files, 'examples': n,
                  'policy_cross_entropy': sum(losses) / n, 'uniform_cross_entropy': sum(uniform_losses) / n,
                  'policy_agreement': sum(agreement) / n,
                  'value_brier': sum((p - r['win']) ** 2 for p, r in zip(probabilities_value, rows)) / n,
                  'constant_quarter_brier': sum((.25 - r['win']) ** 2 for r in rows) / n,
                  'value_calibration_bins': bins,
                  'value_ece': sum(b['count'] * abs(b['predicted'] - b['observed']) for b in bins) / n})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['generate', 'train', 'evaluate'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=list(SPLITS), default='train')
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--targets', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'generate':
        generate(args.data, args.split, args.count, args.targets)
    elif args.mode == 'train':
        train(args.data, args.output, args.epochs, args.seed)
    else:
        evaluate(args.data, args.split, args.model, args.output)


if __name__ == '__main__':
    main()
