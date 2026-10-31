"""Alternate isolated backend probes and compare semantic output before timing."""
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
        for backend in ('python', 'cython'):
            path = args.output / f'{repeat}-{backend}.json'
            subprocess.run([sys.executable, 'scripts/profile_pipeline.py', '--seed', '1400000',
                            '--output', str(path)], env={**os.environ, 'CATAN_ENGINE': backend},
                           check=True, capture_output=True)
            pair[backend] = json.loads(path.read_text())
        for key in ('games', 'search_decisions'):
            clean = lambda items: [{k: v for k, v in r.items() if k != 'seconds'} for r in items]
            if clean(pair['python'][key]) != clean(pair['cython'][key]):
                raise RuntimeError(f'backend divergence in {key}')
        if any(r['winner'] is None for r in pair['python']['games']):
            raise RuntimeError('incomplete trajectory')
        rows.append({backend: sum(r['seconds'] for r in pair[backend]['search_decisions'])
                     for backend in pair})
    reduction = 1 - sum(r['cython'] for r in rows) / sum(r['python'] for r in rows)
    result = {'replicates': rows, 'search_time_reduction': reduction,
              'equivalent': True, 'performance_pass': reduction >= .20}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
