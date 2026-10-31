from copy import deepcopy

import pytest

from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation
from catan_ai.value_model import value_features, baseline_features, predict


def test_value_features_use_only_the_observers_cards():
    game = Game.new(['a', 'b', 'c', 'd'], 1439000)
    game.players[1].resources[Resource.ORE] = 3
    game.players[1].development = [('monopoly', 0)]
    original = Observation.from_game(game, 0)
    before = deepcopy(original)
    game.players[1].resources[Resource.ORE] = 0
    game.players[1].resources[Resource.GRAIN] = 3
    game.players[1].development = [('victory_point', 0)]
    changed = Observation.from_game(game, 0)
    assert value_features(original) == value_features(changed)
    assert baseline_features(original) == baseline_features(changed)
    assert len(value_features(original)) == 38
    assert original == before


def test_value_prediction_checks_shape_and_stays_finite():
    assert predict([0, 0], [0, 0], [1, 1], [1, 2]) == .5
    assert 0 <= predict([1000], [0], [1], [1]) <= 1
    with pytest.raises(ValueError):
        predict([1], [0], [1], [1, 2])
