"""scripts/cython_run.py environment handling (Codex HOLD @f8f8849)."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cython_run as cr  # noqa: E402


class Calls(list):
    fail = frozenset()


@pytest.fixture
def calls(monkeypatch):
    seen = Calls()

    def fake_run(cmd, cwd=None, env=None, capture_output=False, text=False):
        seen.append((list(cmd), dict(env or {})))
        fail = seen.fail
        name = 'bind' if 'bind_cython_checkpoint.py' in ' '.join(cmd) else ('probe' if '-c' in cmd else 'command')
        return SimpleNamespace(returncode=1 if name in fail else 0, stdout='', stderr='')
    monkeypatch.setattr(cr, 'engine_ready', lambda env: True)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    return seen


def test_explicit_model_is_kept_for_the_command_but_not_the_bind(calls, monkeypatch):
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', '/x/candidate.json')
    assert cr.main(['--', 'python', 'eval.py']) == 0
    bind = next(env for cmd, env in calls if 'bind_cython_checkpoint.py' in ' '.join(cmd))
    command = next(env for cmd, env in calls if cmd == ['python', 'eval.py'])
    assert 'CATAN_PARAMETER_MODEL' not in bind
    assert command['CATAN_PARAMETER_MODEL'] == '/x/candidate.json' and command['CATAN_ENGINE'] == 'cython'


def test_explicit_model_that_does_not_load_is_refused(calls, monkeypatch):
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', '/x/python-bound.json')
    calls.fail = {'probe'}
    with pytest.raises(SystemExit, match='does not load under Cython'):
        cr.main(['--', 'python', 'eval.py'])
    assert not any(cmd == ['python', 'eval.py'] for cmd, _ in calls)


def test_bind_failure_prevents_the_command(calls, monkeypatch):
    monkeypatch.delenv('CATAN_PARAMETER_MODEL', raising=False)
    calls.fail = {'bind'}
    with pytest.raises(SystemExit, match='not bound'):
        cr.main(['--', 'python', 'eval.py'])
    assert not any(cmd == ['python', 'eval.py'] for cmd, _ in calls)


def test_no_explicit_model_runs_with_selection_left_to_make_policy(calls, monkeypatch):
    monkeypatch.delenv('CATAN_PARAMETER_MODEL', raising=False)
    assert cr.main(['--', 'python', 'eval.py']) == 0
    command = next(env for cmd, env in calls if cmd == ['python', 'eval.py'])
    assert 'CATAN_PARAMETER_MODEL' not in command
