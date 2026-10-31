from copy import deepcopy

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.finish import winning_plan
from catan_ai.observation import Observation


def position():
    game = Game.new(['a', 'b', 'c', 'd'], 1467900)
    game.phase, game.turn = 'actions', 8
    player = game.players[0]
    player.cities, player.settlements = {0, 2, 4}, {6, 8, 10}
    player.resources.update({Resource.ORE: 2, Resource.GRAIN: 1, Resource.BRICK: 8})
    return game


def test_proved_multi_trade_win_is_legal_and_private():
    game = position()
    game.players[1].resources[Resource.WOOL] = 2
    altered = deepcopy(game)
    altered.deck.reverse()
    altered.players[1].resources[Resource.WOOL] = 0
    altered.players[1].resources[Resource.LUMBER] = 2
    view = Observation.from_game(game, 0, include_events=False)
    other = Observation.from_game(altered, 0, include_events=False)
    assert view == other
    before = deepcopy(view)
    plan = winning_plan(view, 0)
    assert plan == winning_plan(other, 0)
    assert view == before and len(plan) == 3
    for action in plan:
        assert action in game.legal_actions(0)
        game.apply(0, action)
    assert game.winner == 0


def test_no_proof_preserves_baseline_and_rng():
    game = position()
    game.players[0].resources = {r: 0 for r in Resource}
    view = Observation.from_game(game, 0, include_events=False)
    assert winning_plan(view, 0) == []
    baseline, candidate = [make_policy(name, 19) for name in ('planner_robber', 'planner_finish')]
    assert candidate.choose(view, 0) == baseline.choose(view, 0)
    assert candidate.rng.getstate() == baseline.rng.getstate()
    assert winning_plan(Observation.from_game(position(), 0), 0, node_cap=0) == []


def test_known_bank_shortage_cannot_support_winning_sequence():
    game = position()
    game.players[1].resources[Resource.GRAIN] = 18
    assert winning_plan(Observation.from_game(game, 0), 0) == []


def test_knight_can_win_before_roll():
    game = position()
    game.phase = 'roll'
    game.players[0].settlements = {6, 8}
    game.players[0].development = [('knight', 0)]
    game.players[0].knights = 2
    plan = winning_plan(Observation.from_game(game, 0), 0)
    assert plan == [{'type': 'play_development', 'card': 'knight'}]
    game.apply(0, plan[0])
    assert game.winner == 0
