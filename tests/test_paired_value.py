from copy import deepcopy
from pathlib import Path
import runpy

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation
from catan_ai.paired_value import representation, shortlist


def test_action_features_and_shortlist_ignore_hidden_cards():
    game = Game.new(['a', 'b', 'c', 'd'], 1462900)
    game.phase, game.turn = 'actions', 8
    game.players[0].settlements = {0}
    game.players[0].resources.update({Resource.ORE: 3, Resource.GRAIN: 2, Resource.BRICK: 4})
    game.players[1].resources[Resource.WOOL] = 2
    altered = deepcopy(game)
    altered.deck.reverse()
    altered.players[1].resources[Resource.WOOL] = 0
    altered.players[1].resources[Resource.ORE] = 2
    view, other = (Observation.from_game(g, 0, include_events=False) for g in (game, altered))
    assert view == other
    planner = make_policy('planner_robber', 1)
    baseline = planner.choose(view, 0)
    before = deepcopy(view)
    choices = shortlist(view, baseline, planner)
    assert choices == shortlist(other, baseline, planner)
    assert choices[0]['action'] == baseline
    assert 2 <= len(choices) <= 4
    assert all(len(row['features']) == 143 for row in choices)
    assert view == before
    assert representation(view, baseline, planner) == representation(other, baseline, planner)


def test_paired_terminal_rollout_is_repeatable_and_leaves_source_untouched():
    continuation = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                      'scripts/collect_paired_values.py'))['continuation']
    game = Game.new(['a', 'b', 'c', 'd'], 1462901)
    game.phase, game.turn = 'actions', 8
    p = game.players[0]
    p.cities, p.settlements = {0, 2, 4}, {6, 8, 10}
    p.resources.update({Resource.ORE: 3, Resource.GRAIN: 2})
    before = game.public_state(0), game.rng.getstate(), list(game.deck)
    action = {'type': 'build_city', 'vertex': 6}
    first = continuation(game, action, 0, ['tactical'] * 4, 19)
    assert first == continuation(game, action, 0, ['tactical'] * 4, 19)
    assert first['win'] == 1 and not first['truncated']
    assert (game.public_state(0), game.rng.getstate(), game.deck) == before
