from catan_ai.engine import Board, Game, Resource, Tile
from catan_ai.observation import Observation
from catan_ai.strategic import placement_score


def test_second_placement_values_missing_production():
    game = Game.new(['a', 'b', 'c', 'd'], 10)
    game.board = Board(
        (Tile(0, 0, Resource.LUMBER, 6, (0, 1)),
         Tile(1, 0, Resource.GRAIN, 6, (2,)),
         Tile(2, 0, Resource.ORE, 8, (3,))),
        ((0,0), (1,0), (2,0), (3,0)), (), ((),(),(),()), ())
    game.players[0].settlements.add(0)
    view = Observation.from_game(game, 0)
    assert placement_score(view, 0, 2) > placement_score(view, 0, 1)
    assert placement_score(view, 0, 3) > placement_score(view, 0, 1)
