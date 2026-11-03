"""Board row 4, first bar: does a learned win model rank positions better than the served leaf?

Offline prediction gate, not a strength claim. Self-play games of the served bot
(four copies) are sampled at the start of every own turn (the seat to roll),
labelled by whether that seat went on to win. Each row records:

- ``features``: the 38 observation-only ``value_features`` of that seat's view,
  plus the served leaf for that seat and the largest opponent leaf (both
  computed on the card-counted world the served planner itself uses), so the
  learned model can at least reproduce the leaf and passes only by adding to it;
- ``leaf``: the served build-search leaf ``_value(world, seat)``;
- ``shared``: the served shared score ``V(me) - .5 * max V(opp)``.

``generate`` writes one JSON file per game (resumable, ``open('x')``) under a run
directory. ``evaluate`` fits a fixed logistic model on the train split
(standardized on train only; 50 epochs, learning rate .05/(1+epoch/10), L2 .001,
SGD order seed 7, rows weighted so each game counts equally) and reports
test-split AUC for the learned model, the leaf and the shared score, with a
game-clustered bootstrap (1,000 resamples, seed 11) of learned AUC minus the
better baseline AUC. Every constant is fixed in the predeclared config.

Usage:
    PYTHONPATH=runs/<snapshot>/src CATAN_ENGINE=python .venv/bin/python scripts/value_gate.py \
        generate --config experiments/value-gate.json --split train --root runs/value-gate-v1 --workers 2
    ... evaluate --config experiments/value-gate.json --root runs/value-gate-v1 --output runs/value-gate-v1.json
"""
import argparse
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path
import random
import time

from catan_ai.experiments import source_files


def play(args):
    policy, seed, max_turns, path = args
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.value_model import value_features
    path = Path(path)
    if path.exists():
        return 'exists'
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = [make_policy(policy, seed * 17 + i + 1) for i in range(4)]
    rows, sampled = [], set()
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if game.phase == 'roll' and (game.turn, seat) not in sampled:
            sampled.add((game.turn, seat))
            bot = bots[seat]
            world = bot._world(view)
            leaves = [bot._value(world, s) for s in range(4)]
            rows.append({'turn': game.turn, 'seat': seat,
                         'features': value_features(view) + [leaves[seat] / 100,
                                                             max(leaves[s] for s in range(4) if s != seat) / 100],
                         'leaf': leaves[seat], 'shared': bot._score(world, seat)})
        game.apply(seat, bots[seat].choose(view, seat))
    for row in rows:
        row['won'] = game.winner == row['seat']
    record = {'seed': seed, 'policy': policy, 'winner': game.winner, 'turns': game.turn,
              'truncated': game.winner is None, 'rows': rows}
    with path.open('x') as stream:
        json.dump(record, stream)
    return 'done'


def auc(scores, labels, weights):
    pairs = sorted(zip(scores, labels, weights), key=lambda t: t[0])
    pos = sum(w for _, y, w in pairs if y)
    neg = sum(w for _, y, w in pairs if not y)
    if not pos or not neg:
        return None
    below, total, i = 0.0, 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        tied_pos = sum(w for _, y, w in pairs[i:j] if y)
        tied_neg = sum(w for _, y, w in pairs[i:j] if not y)
        total += tied_pos * (below + .5 * tied_neg)
        below += tied_neg
        i = j
    return total / (pos * neg)


def load(root, config, split):
    lo, n = config['splits'][split]
    games = []
    for seed in range(lo, lo + n):
        games.append(json.loads((Path(root) / split / f'{seed}.json').read_text()))
    return games


def fit(games, cfg):
    rows = [(r['features'], r['won'], 1 / len(g['rows'])) for g in games for r in g['rows'] if g['rows']]
    dim = len(rows[0][0])
    total = sum(w for _, _, w in rows)
    means = [sum(w * x[k] for x, _, w in rows) / total for k in range(dim)]
    scales = [math.sqrt(sum(w * (x[k] - means[k]) ** 2 for x, _, w in rows) / total) or 1. for k in range(dim)]
    data = [([(x[k] - means[k]) / scales[k] for k in range(dim)], y, w * len(games) / total) for x, y, w in rows]
    weights, bias = [0.] * dim, 0.
    rng = random.Random(cfg['sgd_seed'])
    for epoch in range(cfg['epochs']):
        rate = cfg['learning_rate'] / (1 + epoch / 10)
        rng.shuffle(data)
        for x, y, w in data:
            z = bias + sum(a * b for a, b in zip(weights, x))
            p = 1 / (1 + math.exp(-max(-40, min(40, z))))
            g = w * (p - y)
            bias -= rate * g
            weights = [a - rate * (g * b + cfg['l2'] * a) for a, b in zip(weights, x)]
    return {'weights': weights, 'bias': bias, 'means': means, 'scales': scales}


def predict(model, x):
    z = model['bias'] + sum(w * (v - m) / s for w, v, m, s in
                            zip(model['weights'], x, model['means'], model['scales']))
    return 1 / (1 + math.exp(-max(-40, min(40, z))))


def evaluate(root, config, output):
    cfg = config['model']
    train, test = load(root, config, 'train'), load(root, config, 'test')
    truncated = sum(g['truncated'] for g in train + test)
    model = fit([g for g in train if not g['truncated']], cfg)
    test = [g for g in test if not g['truncated']]
    per_game = []
    for g in test:
        w = 1 / len(g['rows']) if g['rows'] else 0
        per_game.append([(predict(model, r['features']), r['leaf'], r['shared'], r['won'], w) for r in g['rows']])

    def metrics(games):
        flat = [row for g in games for row in g]
        labels, weights = [r[3] for r in flat], [r[4] for r in flat]
        return {name: auc([r[i] for r in flat], labels, weights)
                for i, name in ((0, 'learned'), (1, 'leaf'), (2, 'shared'))}
    point = metrics(per_game)
    rng = random.Random(config['bootstrap']['seed'])
    diffs = []
    for _ in range(config['bootstrap']['resamples']):
        sample = [per_game[rng.randrange(len(per_game))] for _ in per_game]
        m = metrics(sample)
        diffs.append(m['learned'] - max(m['leaf'], m['shared']))
    diffs.sort()
    lcb = diffs[int(.025 * len(diffs))]
    gate = config['gate']
    passed = (truncated <= gate['maximum_truncated_games'] and lcb > gate['minimum_auc_gain_lcb'])
    result = {'scope': 'offline prediction gate: AUC on held-out self-play positions, not a strength claim',
              'config': config, 'source': source_files(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'train_games': len(train), 'test_games': len(test), 'truncated_games': truncated,
              'test_rows': sum(len(g) for g in per_game), 'auc': point,
              'auc_gain_vs_best_baseline': point['learned'] - max(point['leaf'], point['shared']),
              'auc_gain_bootstrap_lcb95': lcb, 'pass': passed, 'model': model}
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'source', 'model')}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--config', type=Path, required=True)
    gen.add_argument('--split', choices=('train', 'test'), required=True)
    gen.add_argument('--root', type=Path, required=True)
    gen.add_argument('--workers', type=int, default=1)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--config', type=Path, required=True)
    ev.add_argument('--root', type=Path, required=True)
    ev.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == 'evaluate':
        return evaluate(args.root, config, args.output)
    lo, n = config['splits'][args.split]
    directory = args.root / args.split
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(config['policy'], seed, config['max_turns'], str(directory / f'{seed}.json'))
            for seed in range(lo, lo + n)]
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for status in pool.imap_unordered(play, jobs):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(json.dumps({'split': args.split, 'games': done, 'of': len(jobs),
                                  'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
