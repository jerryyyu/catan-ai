"""Bounded, reproducible pipeline profiling and trajectory equivalence probes."""
import argparse
import cProfile
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import pstats
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


def trajectory(policy, seed, capture=False):
    game = Game.new(['a', 'b', 'c', 'd'], seed)
    bots = [make_policy(policy if s == 0 else 'heuristic', seed * 17 + s + 1) for s in range(4)]
    trace = hashlib.sha256()
    snapshots = []
    marks = iter([5, 20, 45])
    mark = next(marks)
    started = time.perf_counter()
    for _ in range(10000):
        if game.winner is not None:
            break
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if capture and mark is not None and game.turn >= mark and game.phase == 'actions' and len(view.actions) > 1:
            snapshots.append(Observation.from_game(game, seat))
            mark = next(marks, None)
        action = bots[seat].choose(view, seat)
        trace.update(json.dumps([seat, action], sort_keys=True).encode())
        game.apply(seat, action)
    trace.update(repr(game.rng.getstate()).encode())
    trace.update(json.dumps(game.public_state(0), sort_keys=True).encode())
    return {'policy': policy, 'seed': seed, 'trace_sha256': trace.hexdigest(),
            'winner': game.winner, 'turns': game.turn,
            'seconds': time.perf_counter() - started}, snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--seed', type=int, default=1200000)
    args = parser.parse_args()
    rows = []
    positions = []
    for policy in ('heuristic', 'tactical'):
        for seed in (args.seed, args.seed + 1):
            row, snapshots = trajectory(policy, seed, capture=not positions)
            rows.append(row)
            if snapshots:
                positions = snapshots
    decisions = []
    for i, view in enumerate(positions):
        bot = make_policy('mc_mean', args.seed + 100 + i)
        start = time.perf_counter()
        action = bot.choose(view, view.seat)
        diagnostic = deepcopy(bot.diagnostics[-1])
        elapsed = time.perf_counter() - start
        diagnostic.pop('seconds')
        decisions.append({'turn': view.turn, 'action': action, 'diagnostic': diagnostic, 'seconds': elapsed})
    result = {'host': platform.node(), 'process_cap': 1, 'source': source_files(),
              'games': rows, 'search_decisions': decisions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'game_seconds': [r['seconds'] for r in rows],
                      'search_seconds': [r['seconds'] for r in decisions]}, indent=2), flush=True)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(trajectory, 'tactical', 1200000)
        profiler.dump_stats(str(args.output.with_suffix('.game.prof')))
        pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(15)
        profiler = cProfile.Profile()
        view = positions[0]
        profiler.runcall(make_policy('mc_mean', 1200100).choose, view, view.seat)
        profiler.dump_stats(str(args.output.with_suffix('.search.prof')))
        pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(15)


if __name__ == '__main__':
    main()
