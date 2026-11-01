import json

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.improved_planners import P_SEVEN_BEFORE_NEXT_TURN
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def play(names, seed, turns=400):
    game = Game.new([f'{n}-{i}' for i, n in enumerate(names)], seed)
    policies = [make_policy(n, seed * 17 + i + 1) for i, n in enumerate(names)]
    while game.winner is None and game.turn < turns:
        seat = game.current
        view = Observation.for_policy(game, seat, policies[seat])
        action = policies[seat].choose(view, seat)
        assert action in view.actions or action['type'] == 'offer_trade'
        game.apply(seat, action)
    return game


@pytest.mark.parametrize('name', ['planner_discard'])
def test_candidates_play_complete_legal_games(name):
    for seed in (2390000, 2390001):
        assert play([name, 'planner_parameter', 'tactical', 'builder'], seed).winner is not None


def test_discard_value_charges_expected_loss_only_above_seven():
    bot = make_policy('planner_discard', 1)
    base = make_policy('planner_parameter', 1)
    game = Game.new(list('abcd'), 3)
    player = game.players[0]
    player.resources = {r: 0 for r in Resource}
    player.resources[Resource.ORE] = 7
    assert bot._value(game, 0) == pytest.approx(base._value(game, 0))
    player.resources[Resource.ORE] = 9
    hand = 1.5 * (1.6 * 3 + .45 * 6)
    expected = P_SEVEN_BEFORE_NEXT_TURN * 4 * DEFAULT[2] * hand / 9
    assert bot._value(game, 0) == pytest.approx(base._value(game, 0) + .5 * 2 - expected)


def test_discard_choice_keeps_the_more_valuable_card():
    game = Game.new(list('abcd'), 5)
    game.phase, game.robber_actor = 'discard', 1
    game.discard_remaining = {0: 4}
    player = game.players[0]
    player.resources = {r: 0 for r in Resource}
    player.resources.update({Resource.ORE: 3, Resource.GRAIN: 2, Resource.WOOL: 3})
    bot = make_policy('planner_discard', 1)
    view = Observation.from_game(game, 0)
    choice = bot.choose(view, 0)
    assert choice['type'] == 'discard'
    # Wool is the cheapest resource in the leaf, and a third wool adds less than a second grain.
    assert choice['resource'] == 'wool'


@pytest.mark.parametrize('name', ['planner_devfix', 'planner_unified'])
def test_new_candidates_play_complete_legal_games(name):
    for seed in (2390000, 2390001, 2390002):
        assert play([name, 'planner_parameter', 'tactical', 'builder'], seed).winner is not None


def _roll_phase_with(card):
    game = Game.new(list('abcd'), 7)
    game.phase, game.turn, game.current = 'roll', 12, 0
    game.players[0].settlements.add(0)
    game.players[0].development = [(card, 0)]
    return game


@pytest.mark.parametrize('card', ['knight', 'year_of_plenty', 'road_building', 'monopoly'])
def test_devfix_rolls_instead_of_playing_before_the_roll(card):
    game = _roll_phase_with(card)
    view = Observation.from_game(game, 0)
    assert any(a['type'] == 'play_development' for a in view.actions)
    assert make_policy('planner_devfix', 1).choose(view, 0) == {'type': 'roll'}
    assert make_policy('planner_parameter', 1).choose(view, 0)['type'] == 'play_development'


def test_devfix_lets_the_search_use_year_of_plenty():
    game = Game.new(list('abcd'), 7)
    game.phase, game.turn, game.current, game.rolled = 'actions', 12, 0, 6
    player = game.players[0]
    player.settlements.add(0)
    player.resources = {r: 0 for r in Resource}
    player.resources.update({Resource.ORE: 2, Resource.GRAIN: 1})
    player.development = [('year_of_plenty', 0)]
    action = make_policy('planner_devfix', 1).choose(Observation.from_game(game, 0), 0)
    # A city needs one more ore and one more grain: the search plays Year of Plenty for them.
    assert action['type'] == 'play_development' and action['card'] == 'year_of_plenty'
    assert sorted(action['resources']) == ['grain', 'ore']


def _offer_position(wool):
    game = Game.new(list('abcd'), 9)
    game.phase, game.turn, game.current = 'actions', 20, 1
    me = game.players[0]
    me.settlements.add(0)
    me.roads.update({(0, 1), (1, 2)})
    me.resources = {r: 0 for r in Resource}
    me.resources.update({Resource.BRICK: 1, Resource.LUMBER: 1, Resource.WOOL: wool, Resource.GRAIN: 1})
    game.players[1].resources = {r: 2 for r in Resource}
    game.apply(1, {'type': 'offer_trade', 'recipient': 0, 'give': {'ore': 1}, 'receive': {'wool': 1}})
    view = Observation.from_game(game, 0)
    assert {'type': 'accept_trade'} in view.actions
    return game, view


def test_unified_rejects_a_trade_that_breaks_an_affordable_settlement():
    game, view = _offer_position(wool=1)
    probe = Game.new(list('abcd'), 9)
    probe.players[0].settlements.add(0)
    probe.players[0].roads.update({(0, 1), (1, 2)})
    probe.players[0].resources = dict(game.players[0].resources)
    probe.phase, probe.current = 'actions', 0
    assert any(a['type'] == 'build_settlement' for a in probe.legal_actions(0))
    assert make_policy('planner_unified', 1).choose(view, 0) == {'type': 'reject_trade'}
    # The served planner's count rule accepts it.
    assert make_policy('planner_parameter', 1).choose(view, 0) == {'type': 'accept_trade'}


def test_unified_accepts_a_trade_that_keeps_the_settlement_and_adds_ore():
    _, view = _offer_position(wool=2)
    assert make_policy('planner_unified', 1).choose(view, 0) == {'type': 'accept_trade'}


def test_unified_picks_monopoly_resource_by_counted_holdings():
    game = Game.new(list('abcd'), 11)
    game.phase, game.turn, game.current, game.rolled = 'actions', 20, 0, 6
    me = game.players[0]
    me.settlements.add(0)
    me.resources = {r: 0 for r in Resource}
    me.development = [('monopoly', 0)]
    # Public bank trades give the opponents known ore.
    for seat in (1, 2, 3):
        game.players[seat].resources = {r: 0 for r in Resource}
        game.players[seat].resources[Resource.WOOL] = 4
    game.current = 1
    for seat in (1, 2, 3):
        game.current = seat
        game.phase = 'actions'
        game.apply(seat, {'type': 'bank_trade', 'give': 'wool', 'receive': 'ore'})
    game.current, game.phase = 0, 'actions'
    view = Observation.from_game(game, 0)
    action = make_policy('planner_unified', 1).choose(view, 0)
    assert action == {'type': 'play_development', 'card': 'monopoly', 'resource': 'ore'}
