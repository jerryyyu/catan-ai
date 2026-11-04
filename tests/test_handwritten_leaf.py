import json
from copy import deepcopy

import pytest

from catan_ai import handwritten_leaf as hw
from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    """A test economic checkpoint bound to the active backend (production binding unchanged)."""
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _midgame(seed=3):
    game = Game.new(list('abcd'), seed)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 16 or game.current != 0 or game.phase != 'actions':
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    return game


def test_building_a_settlement_beats_holding_its_cards():
    bot = make_policy('planner_handwritten_leaf', 1)
    game = next(g for g in map(_midgame, range(3, 40)) if g._legal_settlements(0))
    spot = game._legal_settlements(0)
    me = game.players[0]
    me.resources = {r: 0 for r in Resource} | {Resource.BRICK: 1, Resource.LUMBER: 1, Resource.WOOL: 1, Resource.GRAIN: 1}
    built = deepcopy(game)
    built.apply(0, {'type': 'build_settlement', 'vertex': spot[0]})
    assert bot._leaf(built, 0) > bot._leaf(game, 0)


def test_bank_trade_alone_and_large_hands_cost_value():
    bot = make_policy('planner_handwritten_leaf', 1)
    game = _midgame()
    me = game.players[0]
    me.resources = {r: 0 for r in Resource} | {Resource.WOOL: 4}
    traded = deepcopy(game)
    traded.players[0].resources[Resource.WOOL] = 0
    traded.players[0].resources[Resource.ORE] = 1
    assert bot._leaf(traded, 0) < bot._leaf(game, 0)
    eight = deepcopy(game)
    eight.players[0].resources[Resource.WOOL] = 8
    nine = deepcopy(game)
    nine.players[0].resources[Resource.WOOL] = 9
    assert bot._leaf(nine, 0) - bot._leaf(eight, 0) == pytest.approx(hw.CARD - hw.OVER_SEVEN)


def test_handwritten_leaf_ignores_opponents_hidden_hands():
    game = _midgame(4)
    view = Observation.for_policy(game, 0, make_policy('planner_handwritten_leaf', 1))
    base = make_policy('planner_handwritten_leaf', 1).choose(view, 0)
    for seat in (1, 2, 3):
        hand = game.players[seat].resources
        hand[Resource.WOOL], hand[Resource.ORE] = hand[Resource.ORE], hand[Resource.WOOL]
    view = Observation.for_policy(game, 0, make_policy('planner_handwritten_leaf', 1))
    assert make_policy('planner_handwritten_leaf', 1).choose(view, 0) == base
