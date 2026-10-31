import socket
import subprocess
import sys
import time

from catan_ai.engine import Game
from catan_ai.external_policy import ExternalSearchBot
from catan_ai.observation import Observation


def test_unresponsive_external_process_is_killed_and_failure_recorded(monkeypatch):
    bot = ExternalSearchBot(42, seconds=.05)
    launched = []

    def start_unresponsive_worker():
        parent, child = socket.socketpair()
        bot.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                       pass_fds=(child.fileno(),))
        launched.append(bot.process)
        child.close()
        bot.connection = parent

    monkeypatch.setattr(bot, '_start', start_unresponsive_worker)
    game = Game.new(['a', 'b', 'c', 'd'], 1400071)
    view = Observation.from_game(game, 0)
    started = time.perf_counter()
    action = bot.choose(view, 0)
    assert time.perf_counter() - started < .5
    assert action in view.actions
    assert bot.diagnostics[-1]['fallback']
    assert bot.diagnostics[-1]['budget_failure']
    assert launched[0].poll() is not None
    assert bot.process is None


def test_missing_external_install_is_explicit_not_silent(tmp_path):
    bot = ExternalSearchBot(42, python=str(tmp_path / 'missing-python'))
    game = Game.new(['a', 'b', 'c', 'd'], 1400071)
    view = Observation.from_game(game, 0)
    assert bot.choose(view, 0) in view.actions
    assert bot.diagnostics[-1]['fallback']
    assert 'FileNotFoundError' in bot.diagnostics[-1]['error']
