"""Recover each value-data game's true final points by deterministic replay
(experiments/tiles-margin.json, data.final_points).

The value-data producers (``scripts/rank_data.py`` at aec5881 for runs/value100k-v1 and
at 69d3a4a for runs/value100k-v1b, schema 2, ``decision_rate`` 0) stored the winner and
the final turn but not the final points. Each game is deterministic from its seed: four
copies of ``config['policy']`` seeded ``seed*17+i+1`` on ``Game.new([...], seed)``, one
``Observation.for_policy`` / ``choose`` / ``apply`` per step until a winner or
``max_turns``. With ``decision_rate`` 0 the producer's other calls (value-row features on
the bot or a copy of it, the decision sampler) do not touch the game, the bots' choices
or any stream used by them, so they are omitted here; a divergence would show up as a
winner or turn mismatch, which is recorded, never repaired.

Run it with ``PYTHONPATH=<producer snapshot>/src`` (``catan_ai`` must be the producer's
code): the script imports only the standard library and ``catan_ai`` APIs present in both
producer snapshots and on main, and it refuses a ``catan_ai`` whose ``.py`` sources differ
from the root manifest's ``source`` hashes unless ``--allow-source-drift`` is passed (then
the drift is listed in the manifest). Set ``CATAN_ENGINE`` as the producer did if wanted;
the backend is recorded.

Per game (``<out>/games/<seed>.json``, written with ``open('x')``): ``points`` (true final
VP, hidden victory-point cards included), ``public_points``, ``hidden_vp``, replayed
``winner``/``turns``, the stored ones, ``match`` (winner and turns both equal),
``value_rows_match`` (the replay's (turn, seat, point) sequence of value-row positions
equals the stored one; reported, not part of ``match``), and the stored record's SHA-256.
``<out>/manifest.json`` binds the producer commit (``COMMIT`` file next to the snapshot's
``src`` if present), this script's SHA-256, the data root's manifest identity and the
replay source hashes; a rerun into the same ``--out`` must have the same manifest and
skips games already written. ``<out>/summary.json`` counts games, matches and lists the
mismatches.

    CATAN_ENGINE=cython PYTHONPATH=<snapshot>/src python scripts/replay_final_points.py \
        --root runs/value100k-v1 --out runs/value100k-final-points-v1/value100k-v1 --workers 16
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

SCHEMA = 'catan-final-points-v1'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def producer_commit():
    import catan_ai
    path = Path(catan_ai.__file__).resolve().parents[2] / 'COMMIT'
    return path.read_text().strip() if path.exists() else None


def source_drift(declared):
    """``.py`` files whose SHA-256 differs between the running ``catan_ai`` and the root's
    manifest (backend/binary entries are compared separately, not here)."""
    from catan_ai.experiments import source_files
    running = source_files()
    keys = {k for k in set(declared) | set(running) if k.endswith('.py')}
    return sorted(k for k in keys if declared.get(k) != running.get(k)), running


def value_row_keys(game_rows):
    return [[r['turn'], r['seat'], r['point']] for r in game_rows]


def replay_game(seed, policy, max_turns, points):
    """Replay one seeded self-play game; returns the final engine state summary and the
    (turn, seat, point) positions where the producer stored value rows."""
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = [make_policy(policy, seed * 17 + i + 1) for i in range(4)]
    seen, keys = set(), []
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if 'roll' in points and game.phase == 'roll' and (game.turn, seat, 'roll') not in seen:
            seen.add((game.turn, seat, 'roll'))
            keys.append([game.turn, seat, 'roll'])
        action = bots[seat].choose(view, seat)
        if ('end_turn' in points and game.phase == 'actions' and action['type'] == 'end_turn'
                and (game.turn, seat, 'end') not in seen):
            seen.add((game.turn, seat, 'end'))
            keys.append([game.turn, seat, 'end_turn'])
        game.apply(seat, action)
    return {'points': [int(p.points) for p in game.players],
            'public_points': [int(p.public_points) for p in game.players],
            'winner': game.winner, 'turns': game.turn}, keys


def compare(seed, stored, replayed, keys):
    """The per-game output record (pure: also used by the tests on synthetic records)."""
    match = replayed['winner'] == stored['winner'] and replayed['turns'] == stored['turns']
    return {'seed': seed, 'points': replayed['points'], 'public_points': replayed['public_points'],
            'hidden_vp': [a - b for a, b in zip(replayed['points'], replayed['public_points'])],
            'winner': replayed['winner'], 'turns': replayed['turns'],
            'stored_winner': stored['winner'], 'stored_turns': stored['turns'],
            'stored_truncated_game': stored.get('truncated_game'),
            'match': bool(match), 'value_rows_match': keys == value_row_keys(stored.get('value_rows', [])),
            'winner_has_max_points': (replayed['winner'] is not None and replayed['points'][replayed['winner']]
                                      == max(replayed['points']))}


def run_one(job):
    seed, path, out, identity, policy, max_turns, points, manifest_sha = job
    target = Path(out) / 'games' / f'{seed}.json'
    if target.exists():
        done = json.loads(target.read_text())
        if done.get('manifest_sha256') != manifest_sha or done.get('seed') != seed:
            raise ValueError(f'{target} belongs to a different replay manifest')
        return done
    raw = Path(path).read_bytes()
    stored = json.loads(raw)
    if stored.get('seed') != seed or stored.get('identity') != identity:
        raise ValueError(f'{path} does not belong to the root manifest')
    started = time.monotonic()
    replayed, keys = replay_game(seed, policy, max_turns, points)
    result = {**compare(seed, stored, replayed, keys), 'record_sha256': hashlib.sha256(raw).hexdigest(),
              'manifest_sha256': manifest_sha, 'seconds': time.monotonic() - started}
    with target.open('x') as stream:
        json.dump(result, stream)
    return result


def summarize(results):
    results = sorted(results, key=lambda r: r['seed'])
    bad = [{k: r[k] for k in ('seed', 'winner', 'turns', 'stored_winner', 'stored_turns')}
           for r in results if not r['match']]
    return {'games': len(results), 'matches': len(results) - len(bad), 'mismatches': bad,
            'value_rows_mismatches': [r['seed'] for r in results if not r['value_rows_match']],
            'truncated_games': [r['seed'] for r in results if r['winner'] is None],
            'winner_not_max_points': [r['seed'] for r in results
                                      if r['winner'] is not None and not r['winner_has_max_points']],
            'seeds': [results[0]['seed'], results[-1]['seed']] if results else None}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='value-data root (manifest.json + games/)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--split', type=int, nargs=2, metavar=('FROM', 'TO'),
                        help='only seeds FROM <= seed < TO (default: every game in the root)')
    parser.add_argument('--allow-source-drift', action='store_true',
                        help='replay even if catan_ai differs from the producer sources (recorded)')
    args = parser.parse_args(argv)
    started = time.monotonic()
    root_manifest = json.loads((args.root / 'manifest.json').read_text())
    config = root_manifest['config']
    if config.get('schema') != 2 or config.get('decision_rate') != 0.0:
        raise ValueError('only schema-2 roots with decision_rate 0 are replayed by this script')
    drift, running = source_drift(root_manifest['source'])
    if drift and not args.allow_source_drift:
        raise ValueError(f'catan_ai differs from the producer sources: {drift[:5]} '
                         f'({len(drift)} files); run with the producer snapshot on PYTHONPATH')
    from catan_ai.engine import BACKEND
    manifest = {'schema': SCHEMA, 'experiment': 'experiments/tiles-margin.json',
                'producer_commit': producer_commit(), 'manifest_source_revision': root_manifest.get('source_revision'),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'root': args.root.name, 'root_identity_sha256': root_manifest['identity_sha256'],
                'root_config_sha256': root_manifest['config_sha256'], 'policy': config['policy'],
                'max_turns': config['max_turns'], 'value_points': config.get('value_points', ['roll']),
                'engine_backend': BACKEND, 'replay_source_sha256': canonical_sha(running),
                'source_drift': drift}
    manifest_sha = canonical_sha(manifest)
    manifest['manifest_sha256'] = manifest_sha
    (args.out / 'games').mkdir(parents=True, exist_ok=True)
    path = args.out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()).get('manifest_sha256') != manifest_sha:
            raise ValueError(f'{path} was written by a different replay; use a new --out')
    else:
        with path.open('x') as stream:
            json.dump(manifest, stream, indent=1)
    seeds = sorted(int(p.stem) for p in (args.root / 'games').glob('*.json'))
    if args.split:
        seeds = [s for s in seeds if args.split[0] <= s < args.split[1]]
    jobs = [(s, str(args.root / 'games' / f'{s}.json'), str(args.out), root_manifest['identity_sha256'],
             config['policy'], config['max_turns'], manifest['value_points'], manifest_sha) for s in seeds]
    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_one, jobs, chunksize=4):
                results.append(r)
                if len(results) % 500 == 0:
                    print(json.dumps({'games': len(results), 'of': len(jobs),
                                      'seconds': round(time.monotonic() - started)}), flush=True)
    else:
        results = [run_one(j) for j in jobs]
    summary = {**summarize(results), 'manifest_sha256': manifest_sha, 'seconds': time.monotonic() - started,
               'workers': args.workers, 'split': args.split}
    name = args.out / ('summary.json' if not args.split else f'summary-{args.split[0]}-{args.split[1]}.json')
    with name.open('x') as stream:
        json.dump(summary, stream, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'mismatches'} | {'mismatch_count': len(summary['mismatches'])}))
    return summary


if __name__ == '__main__':
    sys.exit(0 if main() is not None else 1)
