import random

import pytest

from catan_ai.engine import Board, Game, IllegalAction


def test_standard_board_topology_and_materials():
    board = Board.random(random.Random(7))
    assert len(board.tiles) == 19
    assert len(board.vertices) == 54
    assert len(board.edges) == 72
    assert sum(tile.resource == "desert" for tile in board.tiles) == 1
    assert sorted(tile.number for tile in board.tiles if tile.number is not None) == [
        2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12
    ]


def test_setup_is_snake_order_and_second_settlement_grants_resources():
    game = Game.new(["a", "b", "c", "d"], 3)
    seen = []
    for _placement in range(8):
        seen.append(game.current)
        settlement = game.legal_actions(game.current)[0]
        game.apply(game.current, settlement)
        road = game.legal_actions(game.current)[0]
        game.apply(game.current, road)
    assert seen == [0, 1, 2, 3, 3, 2, 1, 0]
    assert game.phase == "roll"
    assert any(sum(player.resources.values()) for player in game.players)


def test_wrong_player_and_unknown_action_are_rejected():
    game = Game.new(["a", "b", "c", "d"], 4)
    with pytest.raises(IllegalAction):
        game.apply(1, {"type": "roll"})
    with pytest.raises(IllegalAction):
        game.apply(0, {"type": "teleport"})


def test_public_state_hides_other_resource_cards():
    game = Game.new(["a", "b", "c", "d"], 5)
    state = game.public_state(0)
    assert state["players"][0]["resources"] is not None
    assert all(player["resources"] is None for player in state["players"][1:])


def test_bank_trade_is_available_at_four_to_one():
    game = Game.new(["a", "b", "c", "d"], 6)
    game.phase = "actions"
    game.players[0].resources[next(iter(game.players[0].resources))] = 4
    trades = [a for a in game.legal_actions(0) if a["type"] == "bank_trade"]
    assert len(trades) == 4
    game.apply(0, trades[0])
    assert sum(game.players[0].resources.values()) == 1
