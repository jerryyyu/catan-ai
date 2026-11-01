"""The Colonist advisor's screen-to-engine mapping (no screenshots, no API calls)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from tools.colonist_advisor.reconstruct import (TEMPLATE, edge_at, from_game, tile_index,  # noqa: E402
                                                to_game, vertex_at)

COLORS = ['red', 'blue', 'white', 'orange']


def _played(seed, steps=200):
    game = Game.new(COLORS, seed)
    bots = [make_policy('tactical', seed + i) for i in range(4)]
    for _ in range(steps):
        if game.winner is not None:
            break
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.from_game(game, seat), seat))
    return game


def test_every_hex_corner_and_side_names_a_board_vertex_and_edge():
    vertices = {vertex_at(row, col, c) for row, n in enumerate((3, 4, 5, 4, 3)) for col in range(n)
                for c in ('top', 'top_right', 'bottom_right', 'bottom', 'bottom_left', 'top_left')}
    edges = {edge_at(row, col, s) for row, n in enumerate((3, 4, 5, 4, 3)) for col in range(n)
             for s in ('top_right', 'right', 'bottom_right', 'bottom_left', 'left', 'top_left')}
    assert vertices == set(range(len(TEMPLATE.vertices)))
    assert edges == set(TEMPLATE.edges)


def test_shared_corners_resolve_to_one_vertex():
    # The right corner pair of the first top-row hex is shared with its right neighbour.
    assert vertex_at(0, 0, 'top_right') == vertex_at(0, 1, 'top_left')
    assert vertex_at(0, 0, 'bottom_right') == vertex_at(0, 1, 'bottom_left')
    assert vertex_at(0, 0, 'bottom') == vertex_at(1, 0, 'top_right') == vertex_at(1, 1, 'top_left')
    assert edge_at(0, 0, 'right') == edge_at(0, 1, 'left')
    assert tile_index(2, 2) == next(i for i, t in enumerate(TEMPLATE.tiles) if (t.q, t.r) == (0, 0))


@pytest.mark.parametrize('seed', [5, 6, 7])
def test_a_played_position_survives_the_round_trip(seed):
    game = _played(seed)
    me = 2
    rebuilt, seat = to_game(from_game(game, COLORS, me))
    assert seat == me
    assert [(t.resource, t.number) for t in rebuilt.board.tiles] == [(t.resource, t.number) for t in game.board.tiles]
    assert sorted(rebuilt.board.ports) == sorted(game.board.ports)
    assert rebuilt.robber_tile == game.robber_tile
    for a, b in zip(rebuilt.players, game.players):
        assert (a.settlements, a.cities, a.roads, a.knights) == (b.settlements, b.cities, b.roads, b.knights)
    assert rebuilt.players[me].resources == game.players[me].resources
    assert rebuilt.road_lengths == game.road_lengths
