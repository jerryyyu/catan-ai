from copy import deepcopy

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.engine import Resource
from catan_ai.observation import Observation
from catan_ai.opening import purchase_horizon
import pytest


@pytest.mark.parametrize('name', ['planner_opening', 'planner_purchase_opening'])
def test_opening_completes_real_snake_setup_without_mutating_views(name):
    game = Game.new(['a', 'b', 'c', 'd'], 1427000)
    bots = [make_policy(name, seat) for seat in range(4)]
    while game.phase.startswith('setup'):
        seat = game.current
        view = Observation.from_game(game, seat)
        before = deepcopy(view)
        action = bots[seat].choose(view, seat)
        assert action in view.actions
        assert view == before
        game.apply(seat, action)
    assert game.setup_index == 8
    assert all(len(p.settlements) == len(p.roads) == 2 for p in game.players)
    assert all(sum(p.resources.values()) > 0 for p in game.players)
    view = Observation.from_game(game, 0)
    assert bots[0].choose(view, 0) == make_policy('planner_robber', 0).choose(view, 0)


def test_purchase_horizon_accounts_for_bottlenecks_and_port_conversion():
    zero = {r: 0 for r in Resource}
    rates = {r: 4 for r in Resource}
    cost = {Resource.ORE: 3, Resource.GRAIN: 2}
    assert purchase_horizon(zero, zero, rates, cost) == float('inf')
    hand = dict(zero, **{Resource.ORE: 3, Resource.GRAIN: 2})
    assert purchase_horizon(hand, zero, rates, cost) == 0
    income = dict(zero)
    income[Resource.WOOL] = 36
    assert purchase_horizon(zero, income, rates, cost) == pytest.approx(20, abs=.008)
    rates[Resource.WOOL] = 2
    assert purchase_horizon(zero, income, rates, cost) == pytest.approx(10, abs=.008)
