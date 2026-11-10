"""Choose the fitted-planner checkpoint for the running engine.

The committed checkpoint (checkpoints/planner_parameter.json) is bound to the Python engine. Under the
Cython engine the served bots need the local Cython-bound copy that ``scripts/bind_cython_checkpoint.py``
writes on each machine. ``select_parameter_checkpoint`` points ``CATAN_PARAMETER_MODEL`` at that copy when
the engine is Cython, the variable is unset and the file exists; an explicit choice always wins and the
Python engine keeps the committed checkpoint. The server and ``bots.make_policy`` both call it, so screens
and data generation under Cython load the served bots without extra setup.
"""
import os
from pathlib import Path

CYTHON_PARAMETER_CHECKPOINT = Path(__file__).resolve().parents[2] / 'checkpoints' / 'planner_parameter.cython.json'


def select_parameter_checkpoint(environ=None, backend=None, path=CYTHON_PARAMETER_CHECKPOINT):
    """Return the path chosen, or None."""
    environ = os.environ if environ is None else environ
    if backend is None:
        from .engine import BACKEND as backend
    if backend == 'cython' and not environ.get('CATAN_PARAMETER_MODEL') and Path(path).exists():
        environ['CATAN_PARAMETER_MODEL'] = str(path)
        return str(path)
    return None
