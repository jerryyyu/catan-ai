"""Resume the frozen comparison runs serially, then execute the final reader."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from catan_ai.experiments import source_files
from catan_ai.search_eval import model_identity


def commands(plan, calibration, scripts):
    result = []
    for entry in plan['runs']:
        result.append([sys.executable, str(scripts / 'parallel_search_eval.py'),
                       '--config', entry['config_path'], '--output', entry['result'],
                       '--dev-result', entry['dev_result'], '--calibration', str(calibration),
                       '--host', plan['planning']['host'], '--process-cap',
                       str(plan['planning']['process_cap'])])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_bytes = args.plan.read_bytes()
        plan = json.loads(plan_bytes)
        if plan.get('diagnostic_only'):
            raise ValueError('diagnostic family cannot launch confirmatory games')
        scripts = Path(__file__).resolve().parent
        identity = {'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
                    'calibration_sha256': hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                    'scripts': {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
                                for name in ('parallel_search_eval.py', 'read_confirmatory_family.py',
                                             'run_confirmatory_family.py')}}
        identity_path = args.output / 'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('supervisor resume identity changed')
        else:
            with identity_path.open('x') as stream:
                json.dump(identity, stream, indent=2)
        terminal = args.output / 'readout.json'
        if terminal.exists():
            raise FileExistsError('family already read')
        try:
            for entry, command in zip(plan['runs'], commands(plan, args.calibration, scripts), strict=True):
                if source_files() != plan['source'] or model_identity(entry['config']) != plan['checkpoint']:
                    raise ValueError('frozen source or checkpoint changed')
                if json.loads(Path(entry['config_path']).read_text()) != entry['config']:
                    raise ValueError('config differs from family plan')
                if Path(entry['result']).exists():
                    print(json.dumps({'existing_result': entry['result'],
                                      'validation': 'final reader required'}), flush=True)
                    continue
                print(json.dumps({'starting': entry['config']['name'], 'time': time.time()}), flush=True)
                subprocess.run(command, check=True)
            subprocess.run([sys.executable, str(scripts / 'read_confirmatory_family.py'),
                            '--plan', str(args.plan), '--output', str(terminal)], check=True)
            print(json.dumps({'terminal': str(terminal), 'time': time.time()}), flush=True)
        except BaseException:
            print(json.dumps({'stopped': True, 'time': time.time(),
                              'resume': 'same command; completed shards retained'}), flush=True)
            raise


if __name__ == '__main__':
    main()
