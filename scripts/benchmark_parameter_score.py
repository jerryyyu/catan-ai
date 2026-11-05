"""Paired score-optimization benchmark; run from the repository root.

Requires dev dependencies and a separately bound CATAN_PARAMETER_MODEL. Uses the
frozen expression in the parity test, alternating arm order on complete games.
Writes a new receipt exclusively; never points this at an existing run artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import runpy
import time

from catan_ai.bots import make_policy
from catan_ai.engine import BACKEND, Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import ParameterPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = runpy.run_path('tests/test_parameter_score_parity.py')['reference_value']
    optimized = ParameterPlanner._value
    config = {'seeds': [2800020, 2800027], 'policy': 'planner_offer_accept',
              'processes': 1, 'backend': BACKEND, 'order': 'alternating',
              'max_turns': 1000, 'bootstrap_seed': 271}
    rows = []
    try:
        for seed in range(config['seeds'][0], config['seeds'][1] + 1):
            arms = [('reference', original), ('optimized', optimized)]
            for name, fn in (arms if seed % 2 == 0 else arms[::-1]):
                ParameterPlanner._value = fn
                start = time.perf_counter()
                game = Game.new(list('abcd'), seed)
                bots = [make_policy(config['policy'], seed * 17 + i + 1) for i in range(4)]
                digest, steps = hashlib.sha256(), 0
                while game.winner is None and game.turn < config['max_turns']:
                    seat = game.current
                    view = Observation.for_policy(game, seat, bots[seat])
                    action = bots[seat].choose(view, seat)
                    digest.update(json.dumps([seat, view.actions, action], sort_keys=True).encode())
                    game.apply(seat, action)
                    steps += 1
                digest.update(json.dumps([game.winner, game.turn, repr(game.rng.getstate())]).encode())
                rows.append(dict(seed=seed, arm=name, seconds=time.perf_counter() - start,
                                 winner=game.winner, turns=game.turn, truncated=game.winner is None,
                                 steps=steps, trace=digest.hexdigest()))
    finally:
        ParameterPlanner._value = optimized
    ratios = []
    for seed in range(config['seeds'][0], config['seeds'][1] + 1):
        a, b = [r for r in rows if r['seed'] == seed]
        assert a['trace'] == b['trace'], f'trajectory differs at seed {seed}'
        ref = next(r for r in (a, b) if r['arm'] == 'reference')
        opt = next(r for r in (a, b) if r['arm'] == 'optimized')
        ratios.append(ref['seconds'] / opt['seconds'])
    rng = random.Random(config['bootstrap_seed'])
    draws = sorted(sum(rng.choices(ratios, k=len(ratios))) / len(ratios) for _ in range(10000))
    result = dict(config=config, reference_revision='eb215a99a5dfb42d05dc293ae8e942144512fc82',
                  source=source_files(), host=platform.node(), games=rows,
                  mean_paired_speedup=sum(ratios) / len(ratios), bootstrap95=[draws[250], draws[9750]],
                  truncations=sum(r['truncated'] for r in rows),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('games', 'source')}, indent=2))


if __name__ == '__main__':
    main()
