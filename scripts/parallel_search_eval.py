"""Run bounded, resumable board shards through the existing evaluation CLI."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from catan_ai.experiments import source_files
from catan_ai.search_eval import model_identity, summarize


def shard_config(config, index):
    if config['stage'] not in ('diagnostic', 'dev', 'heldout'):
        raise ValueError('unsupported evaluation stage for these shards')
    if not 0 <= index < config['seeds_per_panel']:
        raise ValueError('shard index out of range')
    if config['stage'] == 'heldout' and not config.get('requires_dev_pass'):
        raise ValueError('held-out shards require DEV admission')
    result = deepcopy(config)
    result['seeds_per_panel'] = 1
    for panel in result['panels']:
        panel['seed'] += index
    return result


def validate_shard(identity, index, result):
    if result['config'] != shard_config(identity['config'], index):
        raise ValueError('shard config mismatch')
    for field in ('source', 'host'):
        if result[field] != identity[field]:
            raise ValueError(f'shard {field} mismatch')
    if result.get('checkpoint') != identity.get('checkpoint'):
        raise ValueError('shard checkpoint mismatch')
    if result.get('process_cap') != 1:
        raise ValueError('each shard must use the serial evaluator')
    computed = summarize(result['config'], result['games'])
    if computed != result['summary']:
        raise ValueError('shard summary differs from raw games')


def assemble(identity, shards):
    count = identity['config']['seeds_per_panel']
    if set(shards) != set(range(count)):
        raise ValueError('missing or unexpected shards')
    rows = []
    for index in range(count):
        validate_shard(identity, index, shards[index])
        rows.extend(shards[index]['games'])
    order = {p['name']: i for i, p in enumerate(identity['config']['panels'])}
    rows.sort(key=lambda row: (order[row['panel']], row['seed'], row['seat']))
    return {**identity, 'games': rows, 'summary': summarize(identity['config'], rows)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--process-cap', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--dev-result', type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    if platform.node() != args.host:
        raise ValueError('declared host mismatch')
    config = json.loads(args.config.read_text())
    shard_config(config, 0)
    identity = {'config': config, 'source': source_files(), 'host': args.host,
                'process_cap': args.process_cap,
                'runner_sha256': digest(Path(__file__)),
                'dev_sha256': digest(args.dev_result),
                'calibration_sha256': digest(args.calibration)}
    checkpoint = model_identity(config)
    if checkpoint is not None:
        identity['checkpoint'] = checkpoint
    directory = args.output.with_suffix('.shards')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            raise FileExistsError('completed evaluation already exists')
        plan = directory / 'plan.json'
        if plan.exists():
            if json.loads(plan.read_text()) != identity:
                raise ValueError('resume identity differs from original run')
        else:
            with plan.open('x') as stream:
                json.dump(identity, stream, indent=2)
        results = {}
        for index in range(config['seeds_per_panel']):
            result_path = directory / f'{index}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                validate_shard(identity, index, result)
                results[index] = result

        def run(index):
            if source_files() != identity['source'] or model_identity(config) != checkpoint:
                raise RuntimeError('producer changed during evaluation')
            shard = shard_config(config, index)
            path = directory / f'{index}.config.json'
            if path.exists():
                if json.loads(path.read_text()) != shard:
                    raise ValueError('persisted shard config differs')
            else:
                with path.open('x') as stream:
                    json.dump(shard, stream, indent=2)
            command = [sys.executable, '-m', 'catan_ai.search_eval', '--config', str(path),
                       '--output', str(directory / f'{index}.json')]
            for flag, value in (('--dev-result', args.dev_result), ('--calibration', args.calibration)):
                if value is not None:
                    command.extend((flag, str(value)))
            with (directory / f'{index}.log').open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((directory / f'{index}.json').read_text())
            validate_shard(identity, index, result)
            return result

        remaining = iter(i for i in range(config['seeds_per_panel']) if i not in results)
        with ThreadPoolExecutor(max_workers=args.process_cap) as pool:
            pending = {}
            for index in remaining:
                pending[pool.submit(run, index)] = index
                if len(pending) == args.process_cap:
                    break
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    index = pending.pop(future)
                    results[index] = future.result()
                    print(json.dumps({'completed_shard': index, 'completed': len(results),
                                      'total': config['seeds_per_panel']}), flush=True)
                for _ in ready:
                    index = next(remaining, None)
                    if index is not None:
                        pending[pool.submit(run, index)] = index
        if source_files() != identity['source'] or model_identity(config) != checkpoint:
            raise RuntimeError('producer changed during evaluation')
        if digest(args.dev_result) != identity['dev_sha256'] or digest(args.calibration) != identity['calibration_sha256']:
            raise RuntimeError('admission input changed during evaluation')
        result = assemble(identity, results)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
