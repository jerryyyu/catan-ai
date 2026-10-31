"""Exercise full-size aggregation using repeated DEV data, never new games."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / 'scripts/parallel_search_eval.py'))
    plan = json.loads((root / 'experiments/economic-parameters-confirmatory-family-v1.json').read_text())
    plan['diagnostic_only'] = 'Repeated DEV payloads; no independent outcomes or held-out boards played.'
    start = time.monotonic()
    for index, entry in enumerate(plan['runs']):
        dev = json.loads(Path(entry['dev_result']).read_text())
        config = entry['config']
        config['name'] = 'SYNTHETIC-RESOURCE-REHEARSAL-' + config['control']
        for panel_index, panel in enumerate(config['panels']):
            panel['seed'] = 900000000 + index * 100000 + panel_index * 10000
        identity = {'config': config, 'source': plan['source'], 'checkpoint': plan['checkpoint'],
                    'host': dev['host'], 'process_cap': 4, 'diagnostic_only': True}
        shards = {}
        for offset in range(config['seeds_per_panel']):
            shard_config = runner['shard_config'](config, offset)
            rows = []
            for panel, dev_panel in zip(shard_config['panels'], dev['config']['panels'], strict=True):
                for row in dev['games']:
                    if row['panel'] == panel['name'] and row['seed'] == dev_panel['seed'] + offset % 16:
                        duplicate = copy.deepcopy(row)
                        duplicate['seed'] = panel['seed']
                        rows.append(duplicate)
            shards[offset] = {**identity, 'config': shard_config, 'process_cap': 1, 'games': rows,
                              'summary': runner['summarize'](shard_config, rows)}
        result = runner['assemble'](identity, shards)
        path = args.output / f'SYNTHETIC-{config["control"]}.json'
        with path.open('x') as stream:
            json.dump(result, stream, indent=2)
        entry['result'] = str(path.resolve())
        del result, shards, rows, duplicate, dev
        print(json.dumps({'diagnostic_control': config['control'], 'bytes': path.stat().st_size,
                          'elapsed_seconds': time.monotonic() - start}), flush=True)
    with (args.output / 'SYNTHETIC-plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)


if __name__ == '__main__':
    main()
