"""Run a command (a screen, data generation, a ladder) on the Cython engine, the default for new runs.

    uv run --group native --with numpy python scripts/cython_run.py -- python scripts/parallel_search_eval.py --config ...
    # on a host snapshot:  .venv/bin/python $S/scripts/cython_run.py -- .venv/bin/python $S/scripts/rank_data.py ...

Steps, each failing closed:

1. Build the native extensions in place (``scripts/build_native.py build_ext --inplace``) unless the
   compiled engine already matches this source (the engine's own manifest check under CATAN_ENGINE=cython).
2. Bind the local Cython checkpoint (``scripts/bind_cython_checkpoint.py``): it writes only if the
   compatibility games replay identically under Cython and the served bots' traces match Python.
3. Run the command with ``CATAN_ENGINE=cython``. ``bots.make_policy`` then loads the Cython-bound
   checkpoint, and every receipt records the backend and binary hashes through ``source_files()``.

An explicit ``CATAN_PARAMETER_MODEL`` is kept for the command (only the served-checkpoint bind step runs
without it) and must load under Cython, or the launcher refuses before running anything (Codex HOLD
@f8f8849: a custom-model evaluation must never silently fall back to the served weights).

The command runs from the source root of this script. Its exit status is returned.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def engine_ready(env):
    probe = subprocess.run([sys.executable, '-c', 'from catan_ai.engine import BACKEND; print(BACKEND)'],
                           env=env, capture_output=True, text=True, cwd=ROOT)
    return probe.returncode == 0 and probe.stdout.strip() == 'cython'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv or argv.index('--') == len(argv) - 1:
        raise SystemExit('usage: cython_run.py -- COMMAND [ARGS...]')
    command = argv[argv.index('--') + 1:]
    run_env = dict(os.environ, CATAN_ENGINE='cython')
    run_env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'src'), run_env.get('PYTHONPATH')]))
    env = {k: v for k, v in run_env.items() if k != 'CATAN_PARAMETER_MODEL'}  # build and served bind only
    if not engine_ready(env):
        built = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'build_native.py'), 'build_ext', '--inplace'],
                               cwd=ROOT, env=env)
        if built.returncode or not engine_ready(env):
            raise SystemExit('native build failed or the compiled engine does not match this source')
    bound = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'bind_cython_checkpoint.py')], cwd=ROOT, env=env)
    if bound.returncode:
        raise SystemExit('Cython checkpoint not bound (parity failed); refusing to run on Cython')
    explicit = run_env.get('CATAN_PARAMETER_MODEL')
    if explicit:
        probe = subprocess.run([sys.executable, '-c', 'from catan_ai.parameter_planner import load_parameters; load_parameters()'],
                               cwd=ROOT, env=run_env, capture_output=True, text=True)
        if probe.returncode:
            raise SystemExit(f'CATAN_PARAMETER_MODEL={explicit} does not load under Cython (bind it for this engine); '
                             'refusing to run rather than substitute the served checkpoint')
    print(f'cython_run: CATAN_ENGINE=cython, running {" ".join(command)}', file=sys.stderr, flush=True)
    return subprocess.run(command, cwd=ROOT, env=run_env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
