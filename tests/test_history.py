import copy

import pytest

from catan_ai.engine import Game, IllegalAction, Resource
from catan_ai.observation import Observation


def test_stolen_type_is_known_only_to_participants():
    game = Game.new(['a', 'b', 'c', 'd'], 4)
    game.phase = 'steal'
    game.players[1].settlements.add(game.board.tiles[game.robber_tile].vertices[0])
    game.players[1].resources[Resource.ORE] = 1
    game.apply(0, {'type': 'steal', 'victim': 1})
    for seat in (0, 1):
        assert game.observed_events(seat)[-1]['resource_deltas'] == [
            {'ore': 1}, {'ore': -1}, {}, {}]
    for seat in (2, 3, None):
        event = game.observed_events(seat)[-1]
        assert 'resource_deltas' not in event
        assert event['counts'] == [1, 0, 0, 0]


def test_discard_and_purchase_private_details_do_not_leak():
    game = Game.new(['a', 'b', 'c', 'd'], 4)
    game.phase = 'discard'
    game.discard_remaining = {0: 1}
    game.robber_actor = 0
    game.players[0].resources[Resource.ORE] = 1
    game.apply(0, {'type': 'discard', 'resource': 'ore'})
    assert game.observed_events(0)[0]['resource_deltas'][0] == {'ore': -1}
    assert game.observed_events(1)[0]['action'] == {'type': 'discard'}
    assert 'resource_deltas' not in game.observed_events(1)[0]
    game.phase, game.turn = 'actions', 8
    for r in (Resource.ORE, Resource.GRAIN, Resource.WOOL):
        game.players[0].resources[r] = 1
    game.apply(0, {'type': 'buy_development'})
    assert game.observed_events(0)[-1]['drawn_card'] == game.players[0].development[-1][0]
    assert 'drawn_card' not in game.observed_events(1)[-1]
    assert game.observed_events(1)[-1]['resource_deltas'][0] == {'wool': -1, 'grain': -1, 'ore': -1}
    view = Observation.from_game(game, 0)
    assert view.players[0].development_turns == (8,)
    assert view.players[1].development_turns is None


def test_history_is_detached_and_invalid_actions_are_not_recorded():
    game = Game.new(['a', 'b', 'c', 'd'], 4)
    game.apply(0, game.legal_actions(0)[0])
    before = copy.deepcopy(game.events)
    view = Observation.from_game(game, 0)
    view.events[0]['action']['vertex'] = -1
    view.discard_remaining[1] = 100
    assert game.events == before
    assert not game.discard_remaining
    with pytest.raises(IllegalAction):
        game.apply(0, {'type': 'end_turn'})
    assert game.events == before


def test_simulation_can_disable_history_without_changing_transitions():
    game = Game.new(['a', 'b', 'c', 'd'], 4)
    clone = copy.deepcopy(game)
    clone.record_events = False
    for _ in range(16):
        action = game.legal_actions(game.current)[0]
        clone.apply(clone.current, action)
        game.apply(game.current, action)
    a, b = game.public_state(0), clone.public_state(0)
    a.pop('events')
    b.pop('events')
    assert a == b
    assert clone.events == []
    assert len(game.events) == 16
