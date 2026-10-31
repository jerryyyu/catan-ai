import copy
import pytest
from catan_ai.engine import Game, IllegalAction, Resource


def trading_game():
    game = Game.new(['a', 'b', 'c', 'd'], 3)
    game.phase = 'actions'
    game.players[0].resources[Resource.BRICK] = 3
    game.players[1].resources[Resource.ORE] = 2
    return game


def test_bundle_trade_is_atomic_and_returns_to_turn_owner():
    game = trading_game()
    bank = game.bank
    game.apply(0, {'type': 'offer_trade', 'recipient': 1, 'give': {'brick': 3}, 'receive': {'ore': 2}})
    assert game.current == 1 and game.phase == 'trade_response'
    game.apply(1, {'type': 'accept_trade'})
    assert game.players[0].resources[Resource.ORE] == 2
    assert game.players[1].resources[Resource.BRICK] == 3
    assert game.bank == bank
    assert game.current == 0 and game.phase == 'actions' and game.trade_offer is None


def test_offer_does_not_probe_recipient_hand_and_can_be_rejected():
    game = trading_game()
    game.apply(0, {'type': 'offer_trade', 'recipient': 1, 'give': {'brick': 1}, 'receive': {'wool': 2}})
    assert game.legal_actions(0) == []
    assert game.legal_actions(1) == [{'type': 'reject_trade'}]
    assert game.public_state(0)['players'][1]['resources'] is None
    game.apply(1, {'type': 'reject_trade'})
    assert game.players[0].resources[Resource.BRICK] == 3


@pytest.mark.parametrize('give,receive', [({}, {'ore': 1}), ({'brick': -1}, {'ore': 1}), ({'brick': True}, {'ore': 1}), ({'brick': 4}, {'ore': 1}), ({'brick': 1}, {'brick': 1}), ({'invalid': 1}, {'ore': 1})])
def test_invalid_offers_do_not_mutate(give, receive):
    game = trading_game()
    before = copy.deepcopy(game.public_state(0))
    with pytest.raises(IllegalAction):
        game.apply(0, {'type': 'offer_trade', 'recipient': 1, 'give': give, 'receive': receive})
    assert game.public_state(0) == before
