from copy import deepcopy
from dataclasses import replace

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_planner_executes_a_winning_trade_then_city():
    game = Game.new(['a', 'b', 'c', 'd'], 1419001)
    game.phase, game.turn = 'actions', 8
    p = game.players[0]
    p.cities, p.settlements = {0, 2, 4}, {6, 8, 10}
    p.resources[Resource.ORE], p.resources[Resource.GRAIN] = 3, 1
    p.resources[Resource.BRICK] = 4
    bot = make_policy('planner', 1)
    for _ in range(3):
        if game.winner is not None:
            break
        view = Observation.from_game(game, 0)
        action = bot.choose(view, 0)
        assert action in view.actions
        game.apply(0, action)
    assert game.winner == 0


@pytest.mark.parametrize('policy', ['planner', 'planner_dev_sequence'])
def test_planner_preserves_observation_and_ignores_unobserved_cards(policy):
    game = Game.new(['a', 'b', 'c', 'd'], 1419001)
    game.phase, game.turn = 'actions', 8
    game.players[0].settlements = {0}
    game.players[0].resources[Resource.ORE] = 3
    game.players[0].resources[Resource.GRAIN] = 2
    altered = deepcopy(game)
    altered.deck.reverse()
    altered.players[1].development = [('knight', 0)]
    game.players[1].development = [('victory_point', 0)]
    view = Observation.from_game(game, 0)
    before = deepcopy(view)
    changed = Observation.from_game(altered, 0)
    assert view == changed
    action = make_policy(policy, 1).choose(view, 0)
    assert action == make_policy(policy, 1).choose(changed, 0)
    assert view == before


def test_production_robber_counts_cities_and_penalizes_own_income():
    game = Game.new(['a', 'b', 'c', 'd'], 1419001)
    tile_index = next(i for i, tile in enumerate(game.board.tiles) if tile.number == 6)
    vertex = game.board.tiles[tile_index].vertices[0]
    game.players[0].settlements = {vertex}
    bot = make_policy('planner_robber', 1)
    own = bot._robber_score(Observation.from_game(game, 0), 0, tile_index)
    assert own == -10
    game.players[0].settlements.clear()
    game.players[1].settlements = {vertex}
    settlement = bot._robber_score(Observation.from_game(game, 0), 0, tile_index)
    game.players[1].settlements.clear()
    game.players[1].cities = {vertex}
    city = bot._robber_score(Observation.from_game(game, 0), 0, tile_index)
    assert city > 2 * settlement > 0


def test_development_sequence_trades_for_the_missing_card_then_buys():
    game = Game.new(['a', 'b', 'c', 'd'], 1460900)
    game.phase, game.turn = 'actions', 8
    port_vertices = {v for edge, _ in game.board.ports for v in edge}
    game.players[0].settlements = {next(v for v in range(len(game.board.neighbors)) if v not in port_vertices)}
    game.players[0].resources.update({Resource.BRICK: 7, Resource.ORE: 1, Resource.WOOL: 1})
    view = Observation.from_game(game, 0)
    original = make_policy('planner_robber', 1).choose(view, 0)
    bot = make_policy('planner_dev_sequence', 1)
    action = bot.choose(view, 0)
    assert action['type'] == 'bank_trade' and action['receive'] == 'grain'
    assert original != action
    exhausted = replace(view, events=tuple({'action': {'type': 'buy_development'}} for _ in range(25)))
    assert bot.choose(exhausted, 0) == original
    game.apply(0, action)
    action = bot.choose(Observation.from_game(game, 0), 0)
    assert action == {'type': 'buy_development'}
    game.apply(0, action)
    assert len(game.players[0].development) == 1
