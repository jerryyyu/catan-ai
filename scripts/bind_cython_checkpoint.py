"""Bind a local Cython-engine copy of the served fitted-planner checkpoint.

The served checkpoint (checkpoints/planner_parameter.json) pins every source file and the engine
backend, so under CATAN_ENGINE=cython it refuses to load and the server falls back. A Cython binding
also pins the compiled binaries, which differ per machine and compiler, so it is a local artifact
(checkpoints/planner_parameter.cython.json, gitignored), made on each host after building the
extensions:

    uv run --group native python scripts/build_native.py build_ext --inplace
    CATAN_ENGINE=cython uv run --with numpy python scripts/bind_cython_checkpoint.py

It keeps the committed weights and compatibility plan, records the Cython source binding, and writes
only if (1) the checkpoint's compatibility games replay under Cython with exactly the recorded trace
digest (the same check that guards every Python rebind), and (2) seeded games of the served bots
produce identical action traces under both engines (``scripts/trace_parity.py``, Python run in a
subprocess with the committed checkpoint). A receipt with both checks is written next to it. The
server picks the Cython checkpoint up automatically when it runs under CATAN_ENGINE=cython and
CATAN_PARAMETER_MODEL is unset (``catan_ai.api``).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'checkpoints' / 'planner_parameter.cython.json'
PARITY = ROOT / 'scripts' / 'trace_parity.py'
TILES = ROOT / 'checkpoints' / 'value_tiles_v1.npz'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def traces(engine, policies, seeds, model=None):
    env = {k: v for k, v in os.environ.items() if k != 'CATAN_PARAMETER_MODEL'}
    env['CATAN_ENGINE'] = engine
    if model:
        env['CATAN_PARAMETER_MODEL'] = str(model)
    out = subprocess.run([sys.executable, str(PARITY), ','.join(policies), ','.join(map(str, seeds))],
                         env=env, capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        raise SystemExit(f'{engine} parity run failed: {out.stderr[-2000:]}')
    result = json.loads(out.stdout.strip().splitlines()[-1])
    return {p: {k: v for k, v in r.items() if k != 'seconds'} for p, r in result.items()}, \
        {p: r['seconds'] for p, r in result.items()}


def main(argv=None):
    from catan_ai.engine import BACKEND
    from catan_ai.experiments import source_files
    from catan_ai.parameter_planner import SERVED_CHECKPOINT, compatibility_digest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', default='4100001,4100002,4100003,4100004')
    args = parser.parse_args(argv)
    if BACKEND != 'cython':
        raise SystemExit('run with CATAN_ENGINE=cython after building the native extensions')
    data = json.loads(SERVED_CHECKPOINT.read_text())
    candidate = dict(data, source=source_files())
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
        json.dump(candidate, handle, indent=1)
        handle.write('\n')
        temporary = handle.name
    try:
        os.environ['CATAN_PARAMETER_MODEL'] = temporary
        digest = compatibility_digest(candidate['compatibility']['plan'])
        expected = data['compatibility']['trace_sha256']
        if digest != expected:
            raise SystemExit(f'Cython compatibility traces differ: {digest} != {expected}; not bound')
        policies = ['planner_offer_accept'] + (['planner_pv2_tiles'] if TILES.exists() else [])
        seeds = [int(s) for s in args.seeds.split(',')]
        py, py_s = traces('python', policies, seeds)
        cy, cy_s = traces('cython', policies, seeds, temporary)
        if py != cy:
            raise SystemExit(f'Cython traces differ from Python: {py} vs {cy}; not bound')
    finally:
        os.unlink(temporary)
    args.out.write_text(json.dumps(candidate, indent=1) + '\n')
    receipt = {'checkpoint': str(args.out), 'checkpoint_sha256': sha256(args.out),
               'python_checkpoint_sha256': sha256(SERVED_CHECKPOINT), 'weights_identical': True,
               'compatibility_trace_sha256': digest, 'parity_policies': policies, 'parity_seeds': seeds,
               'parity_traces': cy, 'seconds_python': py_s, 'seconds_cython': cy_s,
               'engine_binary_sha256': candidate['source'].get('engine_binary'),
               'scoring_binary_sha256': candidate['source'].get('scoring_binary')}
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('compatibility_trace_sha256', 'parity_policies', 'seconds_python',
                                               'seconds_cython')}, indent=1))
    print(f'bound {args.out}')


if __name__ == '__main__':
    main()
