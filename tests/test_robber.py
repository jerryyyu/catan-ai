import pytest

from catan_ai.engine import Game, IllegalAction, Resource


class Seven:
    def __init__(self):
        self.values = iter([3, 4])

    def randint(self, low, high):
        return next(self.values)


def test_discard_choices_restore_the_rolling_player():
    game = Game.new(['a', 'b', 'c', 'd'], 4)
    game.current = 2
    game.phase = 'roll'
    game.players[0].resources[Resource.BRICK] = 9
    game.players[1].resources[Resource.ORE] = 8
    game.rng = Seven()
    game.apply(2, {'type': 'roll'})
    assert game.phase == 'discard' and game.current == 0
    assert game.public_state(0)['discard_remaining'] == 4
    with pytest.raises(IllegalAction):
        game.apply(2, {'type': 'move_robber', 'tile': 0})
    for _ in range(4):
        game.apply(0, {'type': 'discard', 'resource': 'brick'})
    assert game.current == 1
    for _ in range(4):
        game.apply(1, {'type': 'discard', 'resource': 'ore'})
    assert game.current == 2 and game.phase == 'robber'
    assert game.players[0].resources[Resource.BRICK] == 5
    assert game.players[1].resources[Resource.ORE] == 4


def test_robber_victim_is_selected_and_only_one_card_transfers():
    game = Game.new(['a', 'b', 'c', 'd'], 7)
    tile = next(i for i in range(19) if i != game.robber_tile)
    vertices = game.board.tiles[tile].vertices
    game.players[1].settlements.add(vertices[0])
    game.players[2].settlements.add(vertices[3])
    game.players[1].resources[Resource.BRICK] = 2
    game.players[2].resources[Resource.ORE] = 3
    game.phase = 'robber'
    game.apply(0, {'type': 'move_robber', 'tile': tile})
    assert game.phase == 'steal'
    assert game.legal_actions(0) == [{'type': 'steal', 'victim': 1}, {'type': 'steal', 'victim': 2}]
    game.apply(0, {'type': 'steal', 'victim': 2})
    assert game.players[0].resources[Resource.ORE] == 1
    assert game.players[2].resources[Resource.ORE] == 2
    assert game.players[1].resources[Resource.BRICK] == 2
    assert game.phase == 'actions'
