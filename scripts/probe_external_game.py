"""Complete-game integration probe, not a strength screen or promotion result."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.external_policy import ExternalSearchBot
from catan_ai.observation import Observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1400070)
    parser.add_argument('--seat', type=int, choices=range(4), default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opponents', nargs=3, default=['tactical'] * 3)
    args = parser.parse_args()
    game = Game.new(['a', 'b', 'c', 'd'], args.seed)
    names = list(args.opponents)
    names.insert(args.seat, 'external')
    policies = [make_policy(name, args.seed * 17 + s + 1) for s, name in enumerate(names)]
    policies[args.seat].close()
    external = ExternalSearchBot(args.seed * 17 + args.seat + 1)
    policies[args.seat] = external
    trace, steps, phases = [], [], Counter()
    source = source_files()
    started = time.perf_counter()
    try:
        for step in range(15000):
            if game.winner is not None or game.turn >= 300:
                break
            seat = game.current
            begin = time.perf_counter()
            view = Observation.for_policy(game, seat, policies[seat])
            action = policies[seat].choose(view, seat)
            game.apply(seat, action)
            if seat == args.seat:
                phases[view.phase] += 1
                steps.append(time.perf_counter() - begin)
            trace.append({'seat': seat, 'action': action})
    finally:
        external.close()
    failures = [d for d in external.diagnostics if d['fallback']]
    turn_seconds = Counter()
    for diagnostic in external.diagnostics:
        if not diagnostic['phase'].startswith('setup'):
            turn_seconds[diagnostic['turn']] += diagnostic['seconds']
    unchanged = source == source_files()
    result = {'seed': args.seed, 'external_seat': args.seat, 'opponents': args.opponents, 'winner': game.winner,
              'turns': game.turn, 'truncated': game.winner is None, 'source': source,
              'source_unchanged': unchanged, 'external_identity': external.identity,
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds': time.perf_counter() - started, 'external_step_seconds': steps,
              'phase_counts': dict(phases), 'diagnostics': external.diagnostics, 'trace': trace,
              'external_turn_compute_seconds': dict(turn_seconds),
              'integration_pass': (game.winner is not None and not failures and all(t <= 2 for t in steps)
                                   and max(turn_seconds.values(), default=0) <= 10
                                   and unchanged and external.identity is not None),
              'strength_claim': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('winner','turns','seconds','phase_counts','integration_pass')}, indent=2))
    print(json.dumps({'fallbacks': len(failures), 'errors': dict(Counter(d['error'] for d in failures)),
                      'max_step_seconds': max(steps, default=0)}, indent=2))


if __name__ == '__main__':
    main()
