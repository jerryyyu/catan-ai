from copy import deepcopy
import random

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.learned_planner import ValuePlanner, ValueDrawPlanner, load_value_model
from catan_ai.value_model import value_features
from catan_ai.observation import Observation


def test_value_planner_keeps_opening_and_observation_boundary(monkeypatch):
    monkeypatch.setattr('catan_ai.learned_planner.load_value_model', lambda: {
        'weights': [0.] * 38, 'means': [0.] * 38, 'scales': [1.] * 38})
    game = Game.new(['a', 'b', 'c', 'd'], 1443000)
    view = Observation.from_game(game, 0)
    assert ValuePlanner(random.Random(0)).choose(view, 0) == make_policy('planner_robber', 0).choose(view, 0)
    game.phase, game.turn = 'actions', 8
    game.players[0].settlements = {0}
    game.players[0].resources[Resource.ORE] = 3
    game.players[0].resources[Resource.GRAIN] = 2
    view = Observation.from_game(game, 0)
    before = deepcopy(view)
    action = ValuePlanner(random.Random(0)).choose(view, 0)
    game.players[1].development = [('monopoly', 0)]
    altered = Observation.from_game(game, 0)
    game.players[1].development = [('victory_point', 0)]
    changed = Observation.from_game(game, 0)
    assert altered == changed
    assert ValuePlanner(random.Random(0)).choose(altered, 0) == ValuePlanner(random.Random(0)).choose(changed, 0)
    assert view == before and action in view.actions


def test_missing_or_wrong_checkpoint_fails_without_silent_policy_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_value_model(tmp_path / 'absent')
    path = tmp_path / 'bad.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='checkpoint'):
        load_value_model(path)


def test_purchase_credit_matches_expected_learned_draw_contribution(monkeypatch):
    model = {'weights': [i / 38 for i in range(38)], 'means': [0.] * 38, 'scales': [1.] * 38}
    monkeypatch.setattr('catan_ai.learned_planner.load_value_model', lambda: model)
    bot = ValueDrawPlanner(random.Random(0))
    game = Game.new(['a', 'b', 'c', 'd'], 1443002)
    game.phase, game.turn = 'actions', 8
    before = value_features(Observation.from_game(game, 0))
    expected = 0
    for card, count in zip(('victory_point','knight','monopoly','year_of_plenty','road_building'),(5,14,2,2,2)):
        child = deepcopy(game)
        child.players[0].development.append((card, game.turn))
        after = value_features(Observation.from_game(child, 0))
        expected += count / 25 * sum(w * (a-b) for w,a,b in zip(model['weights'],after,before))
    assert bot.purchase_value == pytest.approx(8 + 20 * expected)
