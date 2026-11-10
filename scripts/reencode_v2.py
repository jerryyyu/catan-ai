"""Re-encode stored value data with encoder v2 by deterministic replay.

The value data (``rank_data.py`` value rows) stored only the 40 legacy features.
Each game is deterministic from its seed (served bots seeded ``seed*17+i+1``), so it
is replayed and, at each stored ``point`` row, encoder v2 features are computed from
the seat's view and card-counted world. Every replayed row must match the stored
row exactly (seat, turn and all 40 legacy features); any mismatch fails the game.

Rows are subsampled deterministically (every ``stride``-th row of the chosen point
per game, starting at offset ``seed % stride``) and written as float16 shards with
labels and row keys. Inputs are the per-root manifests and identities; each shard
records its source games' file hashes, the NPZ's SHA-256, its row count and the
game-id range of its rows (Codex HOLD @c32a17a, P1; ``train_value_v2.shards`` verifies
them); ``DONE.json`` lists every shard's NPZ hash. Runs with the snapshot that produced
the data's features (the legacy features are verified, not trusted).
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_guard import producer_provenance  # noqa: E402
from train_value_v2 import file_sha256, shard_stats  # noqa: E402


def replay(args):
    seed, path, identity, point, stride, max_turns, *rest = args
    policy = rest[0] if rest else 'planner_offer_accept'
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
    if record['winner'] is None:
        # A truncated game has no winner, so every seat's label would read 0 (red team D2, 2026-11-09):
        # replay-verified but contributes no rows; counted in the shard sidecar and DONE.json.
        return seed, [], [], [], hashlib.sha256(raw).hexdigest(), True
    return seed, xs, ys, keys, hashlib.sha256(raw).hexdigest(), False


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


def write_shard(output, split, index, chunk, results):
    """Write one shard (``open('x')``) and its sidecar binding the NPZ bytes, rows and games."""
    xs = [x for _, gx, *_ in results for x in gx]
    ys = [y for _, _, gy, *_ in results for y in gy]
    games = [s for s, gx, *_ in results for _ in gx]
    ks = [k for _, _, _, gk, *_ in results for k in gk]
    arrays = {'x': np.asarray(xs, dtype=np.float16), 'y': np.asarray(ys, dtype=np.uint8),
              'game': np.asarray(games, dtype=np.int32), 'leaf': np.asarray([k[2] for k in ks], dtype=np.float32),
              'shared': np.asarray([k[3] for k in ks], dtype=np.float32)}
    name = Path(output) / f'{split}-{index:04d}.npz'
    with name.open('xb') as stream:
        np.savez(stream, **arrays)
    sidecar = {'split': split, 'seeds': [chunk[0], chunk[-1]], **shard_stats(arrays),
               'npz_sha256': file_sha256(name), 'source_sha256': {str(r[0]): r[4] for r in results},
               'truncated_games': sorted(r[0] for r in results if len(r) > 5 and r[5])}
    with name.with_suffix('.json').open('x') as stream:
        json.dump(sidecar, stream)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='directory holding the source roots')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started_provenance = producer_provenance()
    by_seed = {}
    for root in cfg['roots']:
        manifest = json.loads((args.runs / root['root'] / 'manifest.json').read_text())
        if manifest['identity_sha256'] != root['identity_sha256']:
            raise ValueError(f"{root['root']} identity differs from the declaration")
        for path in (args.runs / root['root'] / 'games').glob('*.json'):
            seed = int(path.stem)
            if seed in by_seed:
                raise ValueError(f'seed {seed} present in two roots')
            by_seed[seed] = (str(path), root['identity_sha256'], manifest['config'].get('policy', 'planner_offer_accept'))
    started, written, truncated = time.monotonic(), {}, []
    for split, (lo, n) in cfg['split'].items():
        seeds = list(range(lo, lo + n))
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            raise ValueError(f'{split}: {len(missing)} seeds missing, e.g. {missing[:3]}')
        for shard_start in range(0, n, cfg['shard_games']):
            chunk = seeds[shard_start:shard_start + cfg['shard_games']]
            jobs = [(s, by_seed[s][0], by_seed[s][1], cfg['point'], cfg['stride'], cfg['max_turns'],
                     by_seed[s][2]) for s in chunk]
            with Pool(args.workers) as pool:
                results = pool.map(replay, jobs, chunksize=4)
            name = write_shard(args.output, split, shard_start // cfg['shard_games'], chunk, results)
            sidecar = json.loads(name.with_suffix('.json').read_text())
            written[name.stem] = sidecar['npz_sha256']
            truncated += sidecar['truncated_games']
            print(json.dumps({'split': split, 'shard': name.name, 'rows': sidecar['rows'],
                              'seconds': round(time.monotonic() - started)}), flush=True)
    with (args.output / 'DONE.json').open('x') as stream:
        json.dump({'config': cfg, 'shards': written, 'truncated_games': truncated,
                   'seconds': time.monotonic() - started, 'provenance': started_provenance}, stream)


if __name__ == '__main__':
    main()
