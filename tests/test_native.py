"""The optional backend preserves complete transitions and rejects stale builds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(code, backend, pythonpath=None):
    env = {**os.environ, 'CATAN_ENGINE': backend}
    if pythonpath:
        env['PYTHONPATH'] = str(pythonpath)
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)


def test_forced_native_rejects_missing_or_stale_build(tmp_path):
    package = tmp_path / 'catan_ai'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'engine.py').write_bytes((ROOT / 'src/catan_ai/engine.py').read_bytes())
    (package / '_engine_py.py').write_text('sentinel = 42')
    code = 'from catan_ai.engine import sentinel; print(sentinel)'
    assert run(code, 'python', tmp_path).stdout.strip() == '42'
    assert run(code, 'auto', tmp_path).stdout.strip() == '42'
    assert 'Cython engine unavailable' in run(code, 'cython', tmp_path).stderr
    (package / '_native_manifest.json').write_text(json.dumps({'source_sha256': 'stale'}))
    assert 'source is stale' in run(code, 'cython', tmp_path).stderr


@pytest.mark.parametrize('lineup', [
    ('tactical', 'heuristic', 'builder', 'expander'),
    ('planner_robber', 'tactical', 'builder', 'expander'),
], ids=['existing-policies', 'challenger'])
def test_native_full_trajectory_equivalence(lineup):
    if not (ROOT / 'src/catan_ai/_native_manifest.json').exists():
        if os.environ.get('CATAN_ENGINE') == 'cython':
            pytest.fail('compiled test job requires native build')
        pytest.skip('optional native build absent')
    code = '''
import hashlib, json
from catan_ai.engine import Game
from catan_ai.bots import make_policy
from catan_ai.observation import Observation
digest = hashlib.sha256()
for seed in (1400002, 1400003):
    game = Game.new(['a','b','c','d'], seed)
    bots = [make_policy(name, seed+i) for i,name in enumerate(LINEUP)]
    for step in range(10000):
        states = [game.public_state(i) for i in range(4)]
        digest.update(json.dumps(states,sort_keys=True).encode())
        digest.update(repr(game.rng.getstate()).encode())
        if game.winner is not None:
            break
        seat = game.current
        view = Observation.for_policy(game,seat,bots[seat])
        action = bots[seat].choose(view,seat)
        digest.update(json.dumps([view.actions, action],sort_keys=True).encode())
        game.apply(seat,action)
    assert game.winner is not None
print(digest.hexdigest())
'''
    code = code.replace('LINEUP', repr(lineup))
    python = run(code, 'python')
    native = run(code, 'cython')
    assert python.returncode == 0, python.stderr
    assert native.returncode == 0, native.stderr
    assert python.stdout == native.stdout


def test_native_road_scoring_matches_reference():
    if not (ROOT / 'src/catan_ai/_native_manifest.json').exists():
        if os.environ.get('CATAN_ENGINE') == 'cython':
            pytest.fail('compiled test job requires native build')
        pytest.skip('optional native build absent')
    result = run('''
import random
from catan_ai.engine import Game, longest_road
from catan_ai.scoring import longest_road as reference
board = Game.new(['a','b','c','d'], 1400004).board
rng = random.Random(1400004)
edges = sorted({tuple(sorted((a,b))) for a, neighbors in enumerate(board.neighbors) for b in neighbors})
for _ in range(200):
    roads = set(rng.sample(edges, rng.randrange(16)))
    blocked = set(rng.sample(range(len(board.neighbors)), rng.randrange(5)))
    assert longest_road(roads, blocked) == reference(roads, blocked)
''', 'cython')
    assert result.returncode == 0, result.stderr
