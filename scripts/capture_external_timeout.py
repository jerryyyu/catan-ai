"""Capture the first failure on a known external regression trajectory."""
import json
import argparse
from pathlib import Path
import pickle

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1426003)
    parser.add_argument('--seat', type=int, default=1)
    parser.add_argument('--policies', nargs=4, default=['tactical', 'external', 'tactical', 'tactical'])
    parser.add_argument('--output', type=Path, default=Path('runs/external-timeout-regression-v1'))
    args = parser.parse_args()
    seed, names = args.seed, args.policies
    assert names[args.seat] == 'external'
    game = Game.new(names, seed)
    policies = [make_policy(name, seed * 17 + seat + 1) for seat, name in enumerate(names)]
    directory = args.output
    directory.mkdir(exist_ok=False)
    trace = []
    try:
        for step in range(50000):
            if game.winner is not None or game.turn >= 1000:
                raise RuntimeError('known failure did not recur')
            seat = game.current
            view = Observation.for_policy(game, seat, policies[seat])
            action = policies[seat].choose(view, seat)
            if seat == args.seat and policies[seat].diagnostics[-1]['fallback']:
                with (directory / 'observation.pkl').open('xb') as stream:
                    pickle.dump(view, stream)
                result = {'seed': seed, 'seat': seat, 'step': step, 'turn': game.turn,
                          'policy_seed': seed * 17 + seat + 1, 'source': source_files(),
                          'diagnostics': policies[seat].diagnostics, 'trace': trace}
                (directory / 'context.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({'step': step, 'turn': game.turn,
                                  'failure': policies[seat].diagnostics[-1]}), flush=True)
                return
            trace.append({'seat': seat, 'action': action})
            game.apply(seat, action)
    finally:
        for policy in policies:
            if hasattr(policy, 'close'):
                policy.close()


if __name__ == '__main__':
    main()
