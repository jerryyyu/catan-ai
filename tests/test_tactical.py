import copy
import random

import pytest

from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation
from catan_ai.tactical import TacticalBot, turn_plan


@pytest.mark.parametrize('seed', range(10))
@pytest.mark.parametrize('acquisition', ['plenty', 'trade'])
def test_twenty_winning_turn_sequences(seed, acquisition):
    game = Game.new(['a', 'b', 'c', 'd'], seed)
    game.phase, game.turn = 'actions', 8
    p = game.players[0]
    p.cities, p.settlements = {0, 2, 4}, {6, 8, 10}
    p.resources[Resource.ORE], p.resources[Resource.GRAIN] = 3, 1
    if acquisition == 'plenty':
        p.development = [('year_of_plenty', 0)]
    else:
        p.resources[Resource.BRICK] = 4
    plan = turn_plan(Observation.from_game(game, 0))
    assert 1 < len(plan) <= 3
    for action in plan:
        assert action in game.legal_actions(0)
        game.apply(0, action)
    assert game.winner == 0


def test_tactical_choices_ignore_hidden_state_and_preserve_inputs():
    game = Game.new(['a', 'b', 'c', 'd'], 3)
    game.phase, game.turn = 'actions', 8
    game.players[0].settlements = {0}
    game.players[0].resources[Resource.ORE] = 3
    game.players[0].resources[Resource.GRAIN] = 1
    game.players[0].development = [('year_of_plenty', 0)]
    changed = copy.deepcopy(game)
    changed.players[1].resources[Resource.ORE] = 10
    changed.deck.reverse()
    view = Observation.from_game(game, 0)
    before = copy.deepcopy(view)
    a = TacticalBot(random.Random(1)).choose(view, 0)
    b = TacticalBot(random.Random(1)).choose(Observation.from_game(changed, 0), 0)
    assert a == b and view == before


def test_acquisitions_after_root_are_not_invented_from_unknown_bank():
    game = Game.new(['a', 'b', 'c', 'd'], 3)
    game.phase = 'actions'
    game.players[0].settlements = {0}
    game.players[0].resources[Resource.BRICK] = 12
    plan = turn_plan(Observation.from_game(game, 0))
    assert sum(a['type'] == 'bank_trade' for a in plan) <= 1


def test_opponent_hoarding_cannot_create_a_fictitious_winning_bank_trade():
    game = Game.new(['a', 'b', 'c', 'd'], 1290000)
    game.phase, game.turn = 'actions', 8
    player = game.players[0]
    player.cities, player.settlements = {0, 2, 4}, {6, 8, 10}
    player.resources[Resource.ORE], player.resources[Resource.GRAIN] = 3, 1
    player.resources[Resource.BRICK] = 4
    game.players[1].resources[Resource.GRAIN] = 18
    view = Observation.from_game(game, 0)
    assert game.bank[Resource.GRAIN] == 0
    assert not any(a['type'] == 'bank_trade' and a['receive'] == 'grain' for a in view.actions)
    plan = turn_plan(view)
    assert not plan


def test_immediate_win_is_taken_even_with_an_opponent_at_nine_points():
    game = Game.new(['a', 'b', 'c', 'd'], 1290001)
    game.phase, game.turn = 'actions', 8
    player = game.players[0]
    player.cities, player.settlements = {0, 2, 4}, {6, 8, 10}
    game.players[1].cities, game.players[1].settlements = {20, 22, 24, 26}, {28}
    player.resources[Resource.ORE], player.resources[Resource.GRAIN] = 3, 1
    player.development = [('year_of_plenty', 0)]
    bot = TacticalBot(random.Random(1))
    for _ in range(3):
        if game.winner is not None:
            break
        game.apply(0, bot.choose(Observation.from_game(game, 0), 0))
    assert game.winner == 0
