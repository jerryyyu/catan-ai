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
