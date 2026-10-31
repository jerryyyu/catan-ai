"""Compare actual serial and parallel evaluation paths on diagnostic boards."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from catan_ai.search_eval import summarize


def outcomes(result):
    return [(row['panel'], row['seed'], row['seat'], arm,
             game['seed'], game['winner'], game['turns'], game['points'], game['truncated'])
            for row in result['games'] for arm in ('candidate', 'baseline')
            for game in (row[arm],)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['stage'] != 'diagnostic':
        raise ValueError('this probe cannot open DEV or held-out boards')
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    runner = Path(__file__).with_name('parallel_search_eval.py')
    results, rows = [], []
    for cap in (1, 2):
        path = args.output / f'cap{cap}.json'
        started = time.perf_counter()
        subprocess.run([sys.executable, str(runner), '--config', str(config_path),
                        '--output', str(path), '--host', platform.node(),
                        '--process-cap', str(cap)], check=True)
        elapsed = time.perf_counter() - started
        result = json.loads(path.read_text())
        computed = summarize(config, result['games'])
        if computed != result['summary'] or result['config'] != config:
            raise ValueError('published result differs from the raw-pair aggregate')
        games = [row[arm] for row in result['games'] for arm in ('candidate', 'baseline')]
        steps = [t for game in games for seat in game['step_seconds'] for t in seat]
        turns = [t['seconds'] for game in games for t in game['turn_timings'] if t['complete']]
        finite = all(isinstance(t, (int, float)) and not isinstance(t, bool)
                     and math.isfinite(t) and t >= 0 for t in steps + turns)
        valid = bool(steps and turns and finite and computed['pass']
                     and max(steps) <= 2 and max(turns) <= 10)
        rows.append({'process_cap': cap, 'seconds': elapsed, 'games': len(games),
                     'max_step': max(steps), 'max_turn_wall': max(turns),
                     'pass': valid, 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        results.append(result)
    identical = outcomes(results[0]) == outcomes(results[1])
    same_producer = results[0]['source'] == results[1]['source']
    report = {'waves': rows, 'identical_outcomes': identical, 'same_producer': same_producer,
              'speedup': rows[0]['seconds'] / rows[1]['seconds'],
              'pass': identical and same_producer and all(row['pass'] for row in rows),
              'purpose': 'runner correctness and timing; not strength evidence'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
