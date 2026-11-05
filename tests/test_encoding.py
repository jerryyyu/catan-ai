import json

import pytest

from catan_ai.bots import make_policy
from catan_ai.encoding import encode, size
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _game(seed=5, turns=30):
    game = Game.new(list('abcd'), seed)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.turn < turns:
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    return game, bots[0]


def _encode(game, bot, seat=0):
    view = Observation.for_policy(game, seat, bot)
    return encode(view, bot._world(view))


def test_encoding_has_declared_size_and_sees_the_robber():
    game, bot = _game()
    x = _encode(game, bot)
    assert len(x) == size()
    robber = [x[i * 8 + 7] for i in range(19)]
    assert robber.index(1.) == game.robber_tile and sum(robber) == 1


def test_encoding_ignores_opponents_hidden_cards():
    game, bot = _game()
    base = _encode(game, bot)
    for seat in (1, 2, 3):
        hand = game.players[seat].resources
        order = list(Resource)
        values = [hand[r] for r in order]
        for r, v in zip(order, values[1:] + values[:1]):  # same count, different cards
            hand[r] = v
        game.players[seat].development = [('monopoly', t) for _, t in game.players[seat].development]
    assert _encode(game, bot) == base


def test_encoding_changes_with_public_board_state():
    game, bot = _game()
    base = _encode(game, bot)
    game.robber_tile = (game.robber_tile + 1) % 19
    assert _encode(game, bot) != base


def test_board_symmetries_are_valid_bijections_of_the_board():
    from catan_ai.encoding import symmetry_permutations
    game, bot = _game()
    perms = symmetry_permutations(game.board)
    assert len(perms) == 12 and perms[0] == list(range(size()))
    assert all(sorted(p) == list(range(size())) for p in perms)
    assert len({tuple(p) for p in perms}) == 12
    x = _encode(game, bot)
    for p in perms:
        y = [x[i] for i in p]
        assert sorted(y) == sorted(x)                      # contents preserved
        assert y[19 * 8:] [-(4 * 27 + 20):] == x[-(4 * 27 + 20):]   # player/global block untouched
        assert sum(y[i * 8 + 7] for i in range(19)) == 1    # still exactly one robber tile
    rot = perms[1]
    six = list(range(size()))
    for _ in range(6):
        six = [six[i] for i in rot]
    assert six == list(range(size()))                       # six 60-degree turns = identity


def test_symmetry_preserves_adjacency():
    """A settlement's neighbouring roads stay its neighbours after any symmetry."""
    from catan_ai.encoding import symmetry_permutations, VERTEX_BLOCK, EDGE_BLOCK
    game, _ = _game()
    board = game.board
    perms = symmetry_permutations(board)
    v0, e0 = 19 * 8, 19 * 8 + 54 * VERTEX_BLOCK
    for p in perms:
        inv = {old: new for new, old in enumerate(p)}
        vnew = {v: (inv[v0 + v * VERTEX_BLOCK] - v0) // VERTEX_BLOCK for v in range(54)}
        enew = {e: (inv[e0 + e * EDGE_BLOCK] - e0) // EDGE_BLOCK for e in range(len(board.edges))}
        for e, (a, b) in enumerate(board.edges):
            assert tuple(sorted(board.edges[enew[e]])) == tuple(sorted((vnew[a], vnew[b])))
