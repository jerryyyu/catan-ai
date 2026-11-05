"""Re-encode stored value data with encoder v2 by deterministic replay.

The value data (``rank_data.py`` value rows) stored only the 40 legacy features.
Each game is deterministic from its seed (served bots seeded ``seed*17+i+1``), so it
is replayed and, at each stored ``point`` row, encoder v2 features are computed from
the seat's view and card-counted world. Every replayed row must match the stored
row exactly (seat, turn and all 40 legacy features); any mismatch fails the game.

Rows are subsampled deterministically (every ``stride``-th row of the chosen point
per game, starting at offset ``seed % stride``) and written as float16 shards with
labels and row keys. Inputs are the per-root manifests and identities; each shard
records its source games' file hashes. Runs with the snapshot that produced the
data's features (the legacy features are verified, not trusted).
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np


def replay(args):
    seed, path, identity, point, stride, max_turns = args
    from catan_ai.bots import make_policy
    from catan_ai.encoding import encode
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.value_model import value_features
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if record['seed'] != seed or record['identity'] != identity:
        raise ValueError(f'{path} does not belong to the declared root')
    stored = [r for r in record['value_rows'] if r['point'] == point]
    keep = set(range(seed % stride, len(stored), stride))
    policy = 'planner_offer_accept'
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = [make_policy(policy, seed * 17 + i + 1) for i in range(4)]
    seen, k, xs, ys, keys = set(), 0, [], [], []
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if point == 'roll' and game.phase == 'roll' and (game.turn, seat) not in seen:
            seen.add((game.turn, seat))
            k = _row(k, keep, stored, game, seat, view, bots[seat], xs, ys, keys, encode, value_features)
        action = bots[seat].choose(view, seat)
        if (point == 'end_turn' and game.phase == 'actions' and action['type'] == 'end_turn'
                and (game.turn, seat) not in seen):
            seen.add((game.turn, seat))
            k = _row(k, keep, stored, game, seat, view, bots[seat], xs, ys, keys, encode, value_features)
        game.apply(seat, action)
    if k != len(stored) or game.winner != record['winner']:
        raise ValueError(f'seed {seed}: replay diverged ({k} of {len(stored)} rows)')
    return seed, xs, ys, keys, hashlib.sha256(raw).hexdigest()


def _row(k, keep, stored, game, seat, view, bot, xs, ys, keys, encode, value_features):
    row = stored[k]
    world = bot._world(view)
    leaves = [bot._value(world, s) for s in range(4)]
    legacy = value_features(view) + [leaves[seat] / 100, max(leaves[s] for s in range(4) if s != seat) / 100]
    if row['seat'] != seat or row['turn'] != game.turn or legacy != row['features']:
        raise ValueError(f'stored row {k} (seat {row["seat"]}, turn {row["turn"]}) does not match the replay')
    if k in keep:
        xs.append(encode(view, world))
        ys.append(int(row['won']))
        keys.append((row['turn'], seat, row['leaf'], row['shared']))
    return k + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='directory holding the source roots')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    by_seed = {}
    for root in cfg['roots']:
        manifest = json.loads((args.runs / root['root'] / 'manifest.json').read_text())
        if manifest['identity_sha256'] != root['identity_sha256']:
            raise ValueError(f"{root['root']} identity differs from the declaration")
        for path in (args.runs / root['root'] / 'games').glob('*.json'):
            seed = int(path.stem)
            if seed in by_seed:
                raise ValueError(f'seed {seed} present in two roots')
            by_seed[seed] = (str(path), root['identity_sha256'])
    started = time.monotonic()
    for split, (lo, n) in cfg['split'].items():
        seeds = list(range(lo, lo + n))
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            raise ValueError(f'{split}: {len(missing)} seeds missing, e.g. {missing[:3]}')
        for shard_start in range(0, n, cfg['shard_games']):
            chunk = seeds[shard_start:shard_start + cfg['shard_games']]
            jobs = [(s, *by_seed[s], cfg['point'], cfg['stride'], cfg['max_turns']) for s in chunk]
            with Pool(args.workers) as pool:
                results = pool.map(replay, jobs, chunksize=4)
            xs = [x for _, gx, _, _, _ in results for x in gx]
            ys = [y for _, _, gy, _, _ in results for y in gy]
            games = [s for s, gx, _, _, _ in results for _ in gx]
            name = args.output / f'{split}-{shard_start // cfg["shard_games"]:04d}.npz'
            with name.open('xb') as stream:
                ks = [k for _, _, _, gk, _ in results for k in gk]
                np.savez(stream, x=np.asarray(xs, dtype=np.float16), y=np.asarray(ys, dtype=np.uint8),
                         game=np.asarray(games, dtype=np.int32),
                         leaf=np.asarray([k[2] for k in ks], dtype=np.float32),
                         shared=np.asarray([k[3] for k in ks], dtype=np.float32))
            (name.with_suffix('.json')).write_text(json.dumps(
                {'split': split, 'seeds': [chunk[0], chunk[-1]], 'rows': len(ys),
                 'source_sha256': {str(s): h for s, _, _, _, h in results}}))
            print(json.dumps({'split': split, 'shard': name.name, 'rows': len(ys),
                              'seconds': round(time.monotonic() - started)}), flush=True)
    (args.output / 'DONE.json').write_text(json.dumps({'config': cfg, 'seconds': time.monotonic() - started}))


if __name__ == '__main__':
    main()
