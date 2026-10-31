"""Deadline-limited local external comparator; failures are visible in diagnostics."""
import os
from pathlib import Path
import socket
import subprocess
import time

from .external_worker import read_message, send_message


class ExternalSearchBot:
    needs_history = True

    def __init__(self, seed, *, seconds=2.0, python=None):
        self.seed, self.seconds = seed, seconds
        self.python = python or os.environ.get('CATAN_EXTERNAL_PYTHON', str(
            Path(__file__).resolve().parents[2] / 'runs/external/catanatron-env/bin/python'))
        self.process = self.connection = None
        self.diagnostics = []
        self.external_diagnostics = self.diagnostics
        self.turn = None
        self.turn_seconds = 0.0
        self.identity = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=1)
            self.process = None

    def _start(self):
        parent, child = socket.socketpair()
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
               'PYTHONHASHSEED': '0'}
        try:
            self.process = subprocess.Popen(
                [self.python, '-m', 'catan_ai.external_worker', str(child.fileno()), str(self.seed)],
                pass_fds=(child.fileno(),), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        self.connection = parent

    def choose(self, view, seat):
        started = time.perf_counter()
        if seat != view.seat or not view.actions:
            raise ValueError('external decision needs the acting seat and legal actions')
        if self.turn != view.turn:
            self.turn, self.turn_seconds = view.turn, 0.0
        budget = min(self.seconds, 10.0 - self.turn_seconds)
        error = None
        belief = None
        action = next((a for a in view.actions if a['type'] == 'end_turn'), view.actions[0])
        try:
            if budget <= 0:
                raise TimeoutError('turn compute budget exhausted')
            if self.process is None:
                self._start()
            remaining = budget - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError('worker startup exceeded deadline')
            self.connection.settimeout(remaining)
            send_message(self.connection, (view, self.seed + len(self.diagnostics)))
            remaining = budget - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError('request transmission exceeded deadline')
            self.connection.settimeout(remaining)
            reply = read_message(self.connection, deadline=started + budget)
            if time.perf_counter() - started > budget:
                raise TimeoutError('external decision exceeded deadline')
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            if reply['action'] not in view.actions:
                raise ValueError('external worker returned an illegal action')
            action = reply['action']
            self.identity = reply.get('identity')
            belief = reply.get('belief')
        except (OSError, EOFError, ValueError, RuntimeError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            if isinstance(exc, (OSError, EOFError)):
                self.close()
        elapsed = time.perf_counter() - started
        self.turn_seconds += elapsed
        self.diagnostics.append({'seconds': elapsed, 'phase': view.phase, 'turn': view.turn,
                                 'fallback': error is not None, 'error': error,
                                 'budget_failure': elapsed > budget or budget <= 0,
                                 'backend': 'catanatron-adapted-depth2-guided-solver-v13',
                                 'belief': belief})
        return action

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
