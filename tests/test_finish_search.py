import json

import pytest

from catan_ai import finish_search as fs
from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    """A test checkpoint bound to the active backend's sources (the production binding is unchanged)."""
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def test_affordability_counts_bank_trades():
    city = [0, 0, 0, 2, 3]  # brick, lumber, wool, grain, ore order of Resource
    order = [r.name for r in Resource]
    assert order.index('GRAIN') == 3 and order.index('ORE') == 4
    assert fs._affordable([0, 0, 0, 2, 3], city, [4] * 5)
    assert not fs._affordable([0, 0, 0, 2, 2], city, [4] * 5)
    assert fs._affordable([4, 0, 0, 2, 2], city, [4] * 5)
    assert fs._affordable([2, 0, 0, 2, 2], city, [2, 4, 4, 4, 4])


def test_can_finish_needs_spots_for_settlements():
    hand = [1, 1, 1, 1, 0]
    assert fs._can_finish(hand, [4] * 5, 1, spots=1, settlements=2, cities_left=4, settlements_left=3)
    assert not fs._can_finish(hand, [4] * 5, 1, spots=0, settlements=2, cities_left=4, settlements_left=3)


def _midgame(seed):
    game = Game.new(list('abcd'), seed)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 16 or game.current != 0 or game.phase != 'actions':
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    return game


def test_below_threshold_it_plays_the_served_move():
    game = _midgame(3)
    assert game.players[0].points < fs.FINISH_POINTS
    view = Observation.for_policy(game, 0, make_policy('planner_offer_accept', 1))
    assert make_policy('planner_finish_next', 1).choose(view, 0) == make_policy('planner_offer_accept', 1).choose(view, 0)


def test_finish_choice_is_legal_and_ignores_hidden_hands():
    game = _midgame(4)
    game.players[0].cities = set(list(game.players[0].settlements)[:2])
    game.players[0].settlements -= game.players[0].cities
    view = Observation.for_policy(game, 0, make_policy('planner_finish_next', 1))
    assert view.players[0].public_points >= 4
    for _ in range(4):  # lift to the activation threshold with hidden VP cards
        game.players[0].development.append(('victory_point', 0))
    view = Observation.for_policy(game, 0, make_policy('planner_finish_next', 1))
    base = make_policy('planner_finish_next', 1).choose(view, 0)
    assert base in view.legal_actions(0) or base['type'] == 'offer_trade'  # offers are free-form
    for seat in (1, 2, 3):
        hand = game.players[seat].resources
        hand[Resource.WOOL], hand[Resource.ORE] = hand[Resource.ORE], hand[Resource.WOOL]
    assert make_policy('planner_finish_next', 1).choose(Observation.for_policy(game, 0, make_policy('planner_finish_next', 1)), 0) == base
