import copy
import random
import pytest
from catan_ai.engine import Board, Game, IllegalAction


def test_random_boards_separate_red_numbers_and_reproduce():
    for seed in range(200):
        board = Board.random(random.Random(seed))
        assert board == Board.random(random.Random(seed))
        red = [tile for tile in board.tiles if tile.number in (6, 8)]
        assert len(red) == 4
        for i, tile in enumerate(red):
            assert all(len(set(tile.vertices) & set(other.vertices)) < 2 for other in red[i+1:])


@pytest.mark.parametrize('action', [
    {'type':'build_settlement', 'vertex':True},
    {'type':'build_settlement', 'vertex':0.0},
    {'type':'build_road', 'edge':None},
    {'type':'build_road', 'edge':['0', '1']},
    {'type':'build_road', 'edge':[0]},
    {'type':'build_road', 'edge':[False, 1]},
])
def test_malformed_actions_reject_without_mutating(action):
    game = Game.new(['a', 'b', 'c', 'd'], 1)
    state = copy.deepcopy(game.public_state(0))
    with pytest.raises(IllegalAction):
        game.apply(0, action)
    assert state == game.public_state(0)
