import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.observation import Observation


@pytest.mark.parametrize('card', ['year_of_plenty', 'monopoly', 'road_building'])
def test_economic_card_is_retained_until_after_production(card):
    game = Game.new(['a', 'b', 'c', 'd'], 1461900)
    game.phase, game.turn = 'roll', 4
    game.players[0].settlements = {0}
    game.players[0].development = [(card, 0)]
    view = Observation.from_game(game, 0)
    assert any(a['type'] == 'play_development' for a in view.actions)
    action = make_policy('planner_card_timing', 0).choose(view, 0)
    assert action == {'type': 'roll'}
    game.apply(0, action)
    assert game.players[0].development == [(card, 0)]


def test_knight_precedes_economic_cards_and_claims_available_army():
    game = Game.new(['a', 'b', 'c', 'd'], 1461901)
    game.phase, game.turn = 'roll', 4
    game.players[0].knights = 2
    game.players[0].development = [('year_of_plenty', 0), ('knight', 0)]
    action = make_policy('planner_card_timing', 0).choose(Observation.from_game(game, 0), 0)
    assert action == {'type': 'play_development', 'card': 'knight'}
    game.apply(0, action)
    assert game.largest_army_holder == 0


def test_post_roll_choice_and_rng_match_existing_planner():
    game = Game.new(['a', 'b', 'c', 'd'], 1461902)
    game.phase, game.turn = 'actions', 4
    game.players[0].development = [('year_of_plenty', 0), ('monopoly', 0)]
    view = Observation.from_game(game, 0)
    candidate, control = (make_policy(name, 4) for name in ('planner_card_timing', 'planner_robber'))
    assert candidate.choose(view, 0) == control.choose(view, 0)
    assert candidate.rng.getstate() == control.rng.getstate()
