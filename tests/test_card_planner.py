from copy import deepcopy
from dataclasses import replace

import pytest

from catan_ai.bots import make_policy
from catan_ai.card_planner import resource_estimates
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_hidden_transfer_preserves_counts_and_known_resource_ratios():
    events = [
        {'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 3, 0, 0],
         'resource_deltas': [{}, {'grain': 2, 'ore': 1}, {}, {}]},
        {'seat': 2, 'action': {'type': 'steal', 'victim': 1}, 'counts': [0, 2, 1, 0]},
    ]
    before = deepcopy(events)
    hands = resource_estimates(events)
    assert hands[2][Resource.GRAIN] == pytest.approx(2 / 3)
    assert hands[1][Resource.ORE] == pytest.approx(2 / 3)
    assert [sum(h.values()) for h in hands] == [0, 2, 1, 0]
    assert events == before


def test_monopoly_uses_observed_resources_and_ignores_hidden_state():
    game = Game.new(['a', 'b', 'c', 'd'], 1433000)
    game.phase, game.turn = 'actions', 4
    game.players[0].development = [('monopoly', 0)]
    game.players[1].resources[Resource.GRAIN] = 4
    game.players[2].resources[Resource.ORE] = 2
    view = Observation.from_game(game, 0)
    events = ({'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 4, 2, 0],
               'resource_deltas': [{}, {'grain': 4}, {'ore': 2}, {}]},)
    view = replace(view, events=events)
    before = deepcopy(view)
    action = make_policy('planner_cards', 1).choose(view, 0)
    assert action == {'type': 'play_development', 'card': 'monopoly', 'resource': 'grain'}
    assert view == before
    game.players[1].resources[Resource.GRAIN] = 0
    game.players[1].resources[Resource.LUMBER] = 4
    changed = replace(Observation.from_game(game, 0), events=events)
    assert changed == view
    assert make_policy('planner_cards', 1).choose(changed, 0) == action
