"""Resumable complete-game value corpus and fixed-recipe prediction screen."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.value_model import value_features, baseline_features, predict

SPLITS = {'train': (1440000, 256), 'validation': (1441000, 64)}
ROOT = Path('runs/outcome-value-v1')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, allow_nan=False)


def identity():
    return {**source_files(), 'pipeline': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def generate(split, limit=None):
    source = identity()
    start, count = SPLITS[split]
    for seed in range(start, start + (count if limit is None else min(count, limit))):
        path = ROOT / split / f'{seed}.json'
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['source'] != source or existing['truncated']:
                raise ValueError('existing shard has a different producer or incomplete game')
            continue
        names = ['planner_robber', 'tactical', 'builder', 'expander']
        shift = seed % 4
        names = names[shift:] + names[:shift]
        game = Game.new(names, seed)
        bots = [make_policy(name, seed * 17 + s) for s, name in enumerate(names)]
        rows, sampled = [], set()
        started = time.perf_counter()
        for _ in range(50000):
            if game.winner is not None or game.turn >= 1000:
                break
            if game.turn in (8, 24, 48, 80) and game.turn not in sampled:
                sampled.add(game.turn)
                for seat in range(4):
                    view = Observation.from_game(game, seat, include_events=False)
                    rows.append({'seat': seat, 'turn': game.turn, 'features': value_features(view),
                                 'baseline': baseline_features(view)})
            seat = game.current
            game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
        for row in rows:
            row['win'] = int(row['seat'] == game.winner) if game.winner is not None else None
        if identity() != source:
            raise RuntimeError('producer changed during generation')
        save(path, {'seed': seed, 'policies': names, 'source': source, 'winner': game.winner,
                    'truncated': game.winner is None, 'rows': rows, 'seconds': time.perf_counter() - started})
        if game.winner is None:
            raise RuntimeError('truncated game preserved; corpus gate fails')
        if (seed - start + 1) % 16 == 0 or seed == start:
            print(json.dumps({'split': split, 'games': seed - start + 1, 'last_seconds': time.perf_counter() - started}), flush=True)


def read_split(split):
    start, count = SPLITS[split]
    games = []
    hashes = {}
    for seed in range(start, start + count):
        path = ROOT / split / f'{seed}.json'
        raw = path.read_bytes()
        game = json.loads(raw)
        assert game['seed'] == seed and not game['truncated'] and game['rows']
        games.append(game)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    assert all(g['source'] == games[0]['source'] for g in games)
    return games, hashes


def fit(games, key):
    rows = [row for game in games for row in game['rows']]
    width = len(rows[0][key])
    means = [sum(r[key][j] for r in rows) / len(rows) for j in range(width)]
    scales = [max(.01, math.sqrt(sum((r[key][j] - means[j]) ** 2 for r in rows) / len(rows))) for j in range(width)]
    means[0], scales[0] = 0., 1.
    weights = [0.] * width
    weights[0] = math.log(1 / 3)
    prepared = [([ (x - m) / s for x, m, s in zip(r[key], means, scales)], r['win'], 1 / len(g['rows']))
                for g in games for r in g['rows']]
    rng = random.Random(7)
    for epoch in range(100):
        rng.shuffle(prepared)
        rate = .05 / (1 + epoch / 20)
        for x, y, weight in prepared:
            score = sum(a * b for a, b in zip(weights, x))
            error = 1 / (1 + math.exp(-max(-40, min(40, score)))) - y
            for j in range(width):
                weights[j] -= rate * weight * (error * x[j] + .001 * weights[j] * (j != 0))
    return {'weights': weights, 'means': means, 'scales': scales}


def metrics(games, key, model):
    losses, briers = [], []
    for game in games:
        probs = [predict(model['weights'], model['means'], model['scales'], r[key]) for r in game['rows']]
        losses.append(sum(-r['win'] * math.log(max(p, 1e-12)) - (1-r['win']) * math.log(max(1-p, 1e-12))
                          for r, p in zip(game['rows'], probs)) / len(probs))
        briers.append(sum((r['win']-p)**2 for r,p in zip(game['rows'],probs)) / len(probs))
    return {'games': len(games), 'rows': sum(len(g['rows']) for g in games),
            'log_loss': sum(losses)/len(losses), 'brier': sum(briers)/len(briers)}


def train():
    games, hashes = read_split('train')
    assert games[0]['source'] == identity()
    model = {'schema': 1, 'source': games[0]['source'], 'data': hashes,
             'value': fit(games, 'features'), 'baseline': fit(games, 'baseline')}
    save(ROOT / 'model.json', model)


def evaluate():
    model = json.loads((ROOT / 'model.json').read_text())
    assert model['source'] == identity()
    games, hashes = read_split('validation')
    assert games[0]['source'] == model['source']
    value = metrics(games, 'features', model['value'])
    baseline = metrics(games, 'baseline', model['baseline'])
    result = {'value': value, 'baseline': baseline, 'data': hashes,
              'model_sha256': hashlib.sha256((ROOT/'model.json').read_bytes()).hexdigest(),
              'pass': value['log_loss'] <= .95 * baseline['log_loss'] and value['brier'] <= baseline['brier'],
              'strength_claim': False}
    save(ROOT / 'validation.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='data'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['generate', 'train', 'evaluate'])
    parser.add_argument('--split', choices=list(SPLITS), default='train')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.mode == 'generate': generate(args.split, args.limit)
    elif args.mode == 'train': train()
    else: evaluate()
