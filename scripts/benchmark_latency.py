"""One bounded simulation with separately reported policy, step and turn timing."""
import argparse
import json
import math
from pathlib import Path
import platform
import time

from catan_ai.experiments import source_files
from catan_ai.sim import play_game


def p95(values):
    values = sorted(values)
    return values[math.ceil(.95 * len(values)) - 1] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--candidate', default='mc_mean')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policies = [args.candidate, 'heuristic', 'heuristic', 'heuristic']
    source = source_files()
    started = time.perf_counter()
    result = play_game(policies, args.seed)
    attempts = [d for d in result.search.get('0', [])
                if d['reason'] not in {'unsupported_phase', 'single_candidate'}]
    summary = {
        'policy_attempt_p95': p95(d['seconds'] for d in attempts),
        'step_p95': p95(result.step_seconds[0]),
        'turn_p95': p95(t['seconds'] for t in result.turn_timings if t['owner'] == 0 and t['complete']),
        'attempts': len(attempts),
        'belief_failures': sum(d['reason'] == 'belief_failure' for d in attempts),
        'truncated': result.truncated, 'elapsed': time.perf_counter() - started,
    }
    if source_files() != source:
        raise RuntimeError('source changed during benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'host': platform.node(), 'process_cap': 1, 'source': source,
                   'policies': policies, 'summary': summary, 'game': result.as_dict()}, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
