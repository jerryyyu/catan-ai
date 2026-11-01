"""Differential evidence must not pass missing, reordered or divergent games."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/verify_native_games.py'))['compare_rows']


def rows():
    games = [{'seed': s, 'trace_sha256': str(s), 'truncated': False, 'seconds': 1.0} for s in (1, 2)]
    return {'python': games, 'cython': deepcopy(games)}


def test_differential_comparison_ignores_only_timing():
    data = rows()
    data['cython'][0]['seconds'] = 9.0
    assert compare(data, [1, 2])['pass']
    data['cython'][0]['trace_sha256'] = 'changed hidden state or transition'
    assert compare(data, [1, 2])['mismatched_seeds'] == [1]
    assert not compare(data, [1, 2])['pass']


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'truncated'])
def test_differential_comparison_fails_closed(mutation):
    data = rows()
    if mutation == 'missing':
        data['python'].pop()
        data['cython'].pop()
    elif mutation == 'reordered':
        data['python'].reverse()
        data['cython'].reverse()
    else:
        data['python'][0]['truncated'] = data['cython'][0]['truncated'] = True
    assert not compare(data, [1, 2])['pass']


def test_state_digest_covers_private_cards_and_rng():
    import hashlib
    from catan_ai.engine import Game, Resource
    encode = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/verify_native_games.py'))['encode']
    game = Game.new(['a', 'b', 'c', 'd'], 2400000)
    original = hashlib.sha256(encode(game)).hexdigest()
    game.players[2].resources[Resource.ORE] += 1
    assert hashlib.sha256(encode(game)).hexdigest() != original
    game.players[2].resources[Resource.ORE] -= 1
    assert hashlib.sha256(encode(game)).hexdigest() == original
    game.rng.random()
    assert hashlib.sha256(encode(game)).hexdigest() != original
