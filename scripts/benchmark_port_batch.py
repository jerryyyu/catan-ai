"""Isolated alternating frozen-original/current pipeline comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(3):
        pair = {}
        for arm in ('before', 'after'):
            target = args.output / f'{repeat}-{arm}.json'
            root = 'runs/port-batch-before/src' if arm == 'before' else 'src'
            env = {**os.environ, 'CATAN_ENGINE': 'python', 'PYTHONPATH': str(Path(root).resolve())}
            subprocess.run([sys.executable, 'scripts/profile_pipeline.py', '--seed', '1476000',
                            '--output', str(target)], env=env, check=True, capture_output=True)
            pair[arm] = json.loads(target.read_text())
        for key in ('games', 'search_decisions'):
            clean = lambda values: [{k: v for k, v in row.items() if k != 'seconds'} for row in values]
            if clean(pair['before'][key]) != clean(pair['after'][key]):
                raise RuntimeError('semantic divergence in ' + key)
        if any(g['winner'] is None for g in pair['after']['games']):
            raise RuntimeError('incomplete trajectory')
        rows.append({arm: sum(r['seconds'] for r in pair[arm]['search_decisions']) for arm in pair})
    reduction = 1 - sum(r['after'] for r in rows) / sum(r['before'] for r in rows)
    result = {'replicates': rows, 'search_time_reduction': reduction, 'equivalent': True,
              'performance_pass': reduction >= .15, 'host': pair['after']['host'], 'process_cap': 1}
    with (args.output / 'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
