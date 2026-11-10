"""Counterfactual "end now" value rows (experiments/endnow-v1.json; value-arms S2, board #1).

The value data has no end-of-turn rows where the acting seat could still build a city or settlement (recipe
review, board #1 comment 6771511862): the data policy always builds. ``planner_pv2_tiles`` nevertheless scores
"end the turn now" against every build, so that comparison rests on extrapolation. This producer supplies the
missing support with true labels:

1. replay a declared game record deterministically (the replay must reproduce the record's winner);
2. among the record's actions-phase decisions where ``end_turn`` is legal and a city or settlement build is
   legal but the policy did not end the turn, pick one uniformly (reservoir sampling, seeded per game);
3. at that decision encode the acting seat's view exactly as an end_turn row (encoder v2, ``bot._world``),
   then fork: the seat ends its turn and the game is played out by copies of the same bots (same dice);
4. label = the forked seat won; a fork reaching ``max_turns`` is dropped and counted (red team D2).

Shards use ``reencode_v2.write_shard`` so ``train_value_v2.shards`` verifies them like any other root.

    endnow_data.py --config experiments/endnow-v1.json --runs runs --output runs/endnow-v1-shards --workers 24
"""
import argparse
from copy import deepcopy
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reencode_v2 import write_shard  # noqa: E402
from artifact_guard import producer_provenance as provenance  # noqa: E402

BUILDS = ('build_city', 'build_settlement')


def fork_seed(seed):
    return seed * 104729 + 11


def play(seed, policy, max_turns, winner=None):
    """(row or None, truncated_fork, eligible_count). ``winner`` checks the replay against the record."""
    from catan_ai.bots import make_policy
    from catan_ai.encoding import encode
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = [make_policy(policy, seed * 17 + i + 1) for i in range(4)]
    pick, eligible, chosen = random.Random(fork_seed(seed)), 0, None
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        snapshot = keep = None
        if game.phase == 'actions':
            legal = game.legal_actions(seat)
            if any(a['type'] == 'end_turn' for a in legal) and any(a['type'] in BUILDS for a in legal):
                # reservoir draw made before the policy moves (it may change bot state); copy only on a keep
                keep = pick.random() * (eligible + 1) < 1
                snapshot = (deepcopy(game), deepcopy(bots), seat, game.turn) if keep else ()
        action = bots[seat].choose(view, seat)
        if snapshot is not None and action['type'] != 'end_turn':
            eligible += 1
            if keep:  # uniform over eligible decisions (the k-th replaces with probability 1/k)
                f_game, f_bots, f_seat, f_turn = snapshot
                f_view = Observation.for_policy(f_game, f_seat, f_bots[f_seat])
                x = encode(f_view, f_bots[f_seat]._world(f_view))
                chosen = (f_game, f_bots, f_seat, f_turn, x)
        game.apply(seat, action)
    if winner is not None and game.winner != winner:
        raise ValueError(f'seed {seed}: replay diverged from the record (winner {game.winner} != {winner})')
    if chosen is None:
        return None, False, eligible
    f_game, f_bots, f_seat, f_turn, x = chosen
    f_game.apply(f_seat, {'type': 'end_turn'})
    while f_game.winner is None and f_game.turn < max_turns:
        s = f_game.current
        v = Observation.for_policy(f_game, s, f_bots[s])
        f_game.apply(s, f_bots[s].choose(v, s))
    if f_game.winner is None:
        return None, True, eligible
    return (x, int(f_game.winner == f_seat), (f_turn, f_seat, 0.0, 0.0)), False, eligible


def job(args):
    seed, path, identity, policy, max_turns = args
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if record['seed'] != seed or record['identity'] != identity:
        raise ValueError(f'{path} does not belong to the declared root')
    if record['winner'] is None:  # truncated source game: no verified replay target, no row
        return seed, [], [], [], hashlib.sha256(raw).hexdigest(), True
    row, truncated, _ = play(seed, policy, max_turns, record['winner'])
    if row is None:
        return seed, [], [], [], hashlib.sha256(raw).hexdigest(), truncated
    x, y, key = row
    return seed, [x], [y], [key], hashlib.sha256(raw).hexdigest(), False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if cfg.get('point') != 'endnow':
        raise ValueError('not an endnow declaration')
    args.output.mkdir(parents=True, exist_ok=False)
    started_provenance = provenance()
    by_seed = {}
    for root in cfg['roots']:
        manifest = json.loads((args.runs / root['root'] / 'manifest.json').read_text())
        if manifest['identity_sha256'] != root['identity_sha256']:
            raise ValueError(f"{root['root']} identity differs from the declaration")
        policy = manifest['config'].get('policy', 'planner_offer_accept')
        for path in (args.runs / root['root'] / 'games').glob('*.json'):
            by_seed[int(path.stem)] = (str(path), root['identity_sha256'], policy)
    started, written, dropped = time.monotonic(), {}, []
    for split, (lo, n) in cfg['split'].items():
        seeds = list(range(lo, lo + n))
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            raise ValueError(f'{split}: {len(missing)} seeds missing, e.g. {missing[:3]}')
        for start in range(0, n, cfg['shard_games']):
            chunk = seeds[start:start + cfg['shard_games']]
            jobs = [(s, *by_seed[s], cfg['max_turns']) for s in chunk]
            with Pool(args.workers) as pool:
                results = pool.map(job, jobs, chunksize=2)
            name = write_shard(args.output, split, start // cfg['shard_games'], chunk, results)
            sidecar = json.loads(name.with_suffix('.json').read_text())
            written[name.stem] = sidecar['npz_sha256']
            dropped += sidecar['truncated_games']
            print(json.dumps({'split': split, 'shard': name.name, 'rows': sidecar['rows'],
                              'seconds': round(time.monotonic() - started)}), flush=True)
    if provenance()['source_files'] != started_provenance['source_files']:
        raise ValueError('source changed while producing; not sealing this shard set')
    with (args.output / 'DONE.json').open('x') as stream:
        json.dump({'config': cfg, 'shards': written, 'truncated_games': dropped,
                   'seconds': time.monotonic() - started, 'provenance': started_provenance}, stream)


if __name__ == '__main__':
    main()
