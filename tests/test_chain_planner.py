from copy import deepcopy

from catan_ai.bots import make_policy
from catan_ai.chain_planner import bank_floor, supported
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_two_trades_then_city_wins_without_exposing_opponent_cards():
    game = Game.new(['a', 'b', 'c', 'd'], 1435000)
    game.phase, game.turn = 'actions', 8
    p = game.players[0]
    p.cities, p.settlements = {0, 2, 4}, {6, 8, 10}
    p.resources[Resource.ORE], p.resources[Resource.GRAIN] = 2, 1
    p.resources[Resource.BRICK] = 8
    bot = make_policy('planner_chain', 0)
    actions = []
    for _ in range(3):
        view = Observation.from_game(game, 0)
        before = deepcopy(view)
        action = bot.choose(view, 0)
        assert view == before
        assert action in view.actions
        actions.append(action['type'])
        game.apply(0, action)
    assert actions == ['bank_trade', 'bank_trade', 'build_city']
    assert game.winner == 0


def test_bank_guarantees_do_not_invent_repeat_trade_supply():
    game = Game.new(['a', 'b', 'c', 'd'], 1435001)
    game.phase, game.turn = 'actions', 8
    game.players[0].resources[Resource.BRICK] = 8
    game.players[1].resources[Resource.ORE] = 18
    view = Observation.from_game(game, 0)
    floor = bank_floor(view)
    assert floor[Resource.ORE] == 1
    action = {'type': 'bank_trade', 'give': 'brick', 'receive': 'ore'}
    assert supported(action, floor)
    floor[Resource.ORE] -= 1
    assert not supported(action, floor)
