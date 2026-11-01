"""Seeded full-state differential evidence; timing includes verification overhead.

Run bounded, disjoint seed chunks into new directories. Never overwrite receipts.
This is a correctness probe, not a gameplay-strength or engine-throughput benchmark.
"""
import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, random.Random):
        return canonical(value.getstate())
    if is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(canonical(k)): canonical(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    return value


def encode(value):
    return json.dumps(value, default=canonical, sort_keys=True, separators=(',', ':')).encode() + b'\n'


def trajectory(seed, lineup, max_steps):
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation

    game = Game.new(['a', 'b', 'c', 'd'], seed)
    rotated = lineup[seed % 4:] + lineup[:seed % 4]
    bots = [make_policy(name, seed * 17 + seat) for seat, name in enumerate(rotated)]
    digest = hashlib.sha256()
    coverage = Counter()
    event_count = 0
    started = time.perf_counter()
    steps = 0
    for step in range(max_steps + 1):
        # Hash every field, including private state and RNG. Hash the growing
        # event prefix incrementally; the complete final history is checked too.
        state = {f.name: getattr(game, f.name) for f in fields(game) if f.name != 'events'}
        digest.update(encode(state))
        digest.update(encode(game.events[event_count:]))
        event_count = len(game.events)
        digest.update(encode([game.public_state(seat) for seat in range(4)]))
        digest.update(encode([game.legal_actions(seat) for seat in range(4)]))
        if game.winner is not None or step == max_steps:
            break
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        action = bots[seat].choose(view, seat)
        digest.update(encode([seat, view.actions, action]))
        coverage[action['type']] += 1
        game.apply(seat, action)
        steps += 1
    digest.update(encode(game.events))
    return {'seed': seed, 'lineup': rotated, 'steps': steps, 'turns': game.turn,
            'winner': game.winner, 'truncated': game.winner is None,
            'trace_sha256': digest.hexdigest(), 'actions': dict(coverage),
            'seconds': time.perf_counter() - started}


def compare_rows(rows, expected_seeds):
    """Fail closed on missing/reordered records as well as semantic divergence."""
    complete = all([r['seed'] for r in batch] == expected_seeds for batch in rows.values())
    mismatches = [a['seed'] for a, b in zip(rows['python'], rows['cython'])
                  if {k: v for k, v in a.items() if k != 'seconds'} !=
                  {k: v for k, v in b.items() if k != 'seconds'}]
    truncations = {backend: sum(r['truncated'] for r in batch) for backend, batch in rows.items()}
    return {'pass': complete and not mismatches and not any(truncations.values()),
            'complete': complete, 'mismatched_seeds': mismatches, 'truncations': truncations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--lineup', nargs=4, default=['tactical', 'heuristic', 'builder', 'expander'])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.games < 1 or args.max_steps < 1:
        parser.error('games and max-steps must be positive')
    if args.worker:
        with args.output.open('x') as stream:
            for seed in range(args.seed, args.seed + args.games):
                stream.write(json.dumps(trajectory(seed, args.lineup, args.max_steps)) + '\n')
                stream.flush()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(['git', 'ls-files', 'src', 'scripts/verify_native_games.py'], cwd=ROOT, text=True).splitlines()
    tracked = sorted(set(tracked + ['scripts/verify_native_games.py']))
    config = {'seed': args.seed, 'games': args.games, 'max_steps': args.max_steps, 'lineup': args.lineup,
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked},
              'host': platform.node(), 'python': sys.version, 'process_cap': 1,
              'native_manifest': json.loads((ROOT / 'src/catan_ai/_native_manifest.json').read_text()),
              'bar': 'all seeds complete; identical per-game state/legality/observation/action/RNG hashes',
              'uncertainty_interval': None, 'scope': 'deterministic correctness; no strength or speed claim'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    started = time.perf_counter()
    for backend in ('python', 'cython'):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output',
                   str((args.output / f'{backend}.jsonl').resolve()), '--seed', str(args.seed),
                   '--games', str(args.games), '--max-steps', str(args.max_steps), '--lineup', *args.lineup]
        with (args.output / f'{backend}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, 'CATAN_ENGINE': backend,
                           'PYTHONPATH': str(ROOT / 'src')}, stdout=log, stderr=log, check=True)
    rows = {b: [json.loads(line) for line in (args.output / f'{b}.jsonl').read_text().splitlines()]
            for b in ('python', 'cython')}
    result = compare_rows(rows, list(range(args.seed, args.seed + args.games)))
    stable = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in config['files'].items())
    result.update({'pass': result['pass'] and stable, 'source_unchanged': stable,
                   'games_per_backend': {b: len(r) for b, r in rows.items()},
                   'elapsed_seconds': time.perf_counter() - started,
                   'scope': config['scope'], 'uncertainty_interval': None})
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
