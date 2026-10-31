"""Previously failing board/lineup regression under the revised external producer."""
import argparse
import json
from pathlib import Path
import time

from catan_ai.experiments import source_files
from catan_ai.sim import play_game


CASES = [
    (1455001, ['builder', 'external', 'expander', 'planner_robber']),
    (1454111, ['builder', 'external', 'planner_robber', 'expander']),
    (1450103, ['external', 'builder', 'external', 'expander']),
    (1448110, ['external', 'builder', 'external', 'expander']),
    (1431103, ['planner_robber', 'builder', 'external', 'expander']),
    (1426003, ['tactical', 'external', 'tactical', 'tactical']),
    (1426102, ['external', 'builder', 'external', 'expander']),
    (1422001, ['tactical', 'external', 'tactical', 'tactical']),
    (1422100, ['planner_development', 'builder', 'external', 'expander']),
    (1422100, ['external', 'builder', 'external', 'expander']),
    (1422100, ['builder', 'planner_development', 'external', 'expander']),
    (1422100, ['builder', 'external', 'external', 'expander']),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_files()
    summaries = []
    for index, (seed, names) in enumerate(CASES):
        started = time.perf_counter()
        result = play_game(names, seed, 300).as_dict()
        records = [d for e in result['external'].values() for d in e['diagnostics']]
        failures = [d for d in records if d['fallback'] or d['budget_failure']]
        steps = [t for s in result['external'] for t in result['step_seconds'][int(s)]]
        turns = {}
        for s, e in result['external'].items():
            for d in e['diagnostics']:
                if not d['phase'].startswith('setup'):
                    key = (s, d['turn'])
                    turns[key] = turns.get(key, 0) + d['seconds']
        summary = {'seed': seed, 'policies': names, 'winner': result['winner'],
                   'truncated': result['truncated'], 'failures': len(failures),
                   'errors': sorted({d['error'] for d in failures if d['error']}),
                   'max_step': max(steps), 'max_turn': max(turns.values()),
                   'seconds': time.perf_counter() - started}
        summary['pass'] = not result['truncated'] and not failures and summary['max_step'] <= 2 and summary['max_turn'] <= 10
        (args.output / f'{index}.json').write_text(json.dumps({'summary': summary, 'result': result}, indent=2))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        if source_files() != source:
            raise RuntimeError('source changed during regression')
    (args.output / 'summary.json').write_text(json.dumps({'source': source, 'games': summaries,
                                                       'pass': all(r['pass'] for r in summaries)}, indent=2))


if __name__ == '__main__':
    main()
