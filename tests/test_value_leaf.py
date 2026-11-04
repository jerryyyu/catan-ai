import json
import math

import pytest

from catan_ai import value_leaf
from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_value_leaf_loads_bound_model_and_scores_finitely():
    bot = make_policy('planner_value_leaf', 1)
    game = Game.new(list('abcd'), 5)
    assert math.isfinite(bot._leaf(bot._world(Observation.from_game(game, 0)), 0))


def test_value_leaf_rejects_a_changed_checkpoint(tmp_path):
    data = json.loads(value_leaf.CHECKPOINT.read_text())
    data['model']['bias'] += 1
    path = tmp_path / 'tampered.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        value_leaf.load_value_leaf(path)


def test_value_leaf_ignores_opponents_hidden_hands():
    game = Game.new(list('abcd'), 7)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 12:
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    while game.current != 0 or game.phase != 'actions':
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    view = Observation.for_policy(game, 0, bots[0])
    base = make_policy('planner_value_leaf', 1).choose(view, 0)
    for seat in (1, 2, 3):
        hand = game.players[seat].resources
        hand[Resource.WOOL], hand[Resource.ORE] = hand[Resource.ORE], hand[Resource.WOOL]
    assert make_policy('planner_value_leaf', 1).choose(Observation.for_policy(game, 0, bots[0]), 0) == base


def test_scaled_value_leaf_maps_logit_onto_served_units():
    bot = make_policy('planner_value_leaf_scaled', 1)
    game = bot._world(Observation.from_game(Game.new(list('abcd'), 5), 0))
    s = bot.scale
    assert s['train_games'] == 600 and s['slope'] > 0
    assert bot._leaf(game, 0) == pytest.approx(s['leaf_mean'] + s['slope'] * (bot._logit(game, 0) - s['logit_mean']))


def test_end_turn_value_leaf_uses_its_own_frozen_model_and_scale():
    end, v1 = make_policy('planner_value_leaf_end', 1), make_policy('planner_value_leaf_scaled', 1)
    assert end.value_model != v1.value_model and end.scale['train_rows'] == 45047
    game = end._world(Observation.from_game(Game.new(list('abcd'), 5), 0))
    s = end.scale
    assert end._leaf(game, 0) == pytest.approx(s['leaf_mean'] + s['slope'] * (end._logit(game, 0) - s['logit_mean']))
