import copy
import random

from catan_ai.bots import HeuristicBot
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_hidden_card_permutations_do_not_change_observation_or_decision():
    game = Game.new(['a', 'b', 'c', 'd'], 42)
    game.phase = 'actions'
    game.players[0].resources[Resource.BRICK] = 1
    game.players[1].resources[Resource.ORE] = 1
    game.players[2].resources[Resource.WOOL] = 1
    game.players[1].development = [('knight', 0)]
    game.players[2].development = [('victory_point', 0)]
    changed = copy.deepcopy(game)
    changed.players[1].resources, changed.players[2].resources = changed.players[2].resources, changed.players[1].resources
    changed.players[1].development, changed.players[2].development = changed.players[2].development, changed.players[1].development
    changed.deck.reverse()
    a, b = Observation.from_game(game, 0), Observation.from_game(changed, 0)
    assert a == b
    assert HeuristicBot(random.Random(1)).choose(a, 0) == HeuristicBot(random.Random(1)).choose(b, 0)
    assert not hasattr(a, 'deck') and not hasattr(a, 'rng')
    assert a.players[1].resources is None and a.players[1].development is None


def test_mutating_policy_inputs_does_not_mutate_engine():
    game = Game.new(['a', 'b', 'c', 'd'], 42)
    view = Observation.from_game(game, 0)
    view.players[0].resources[Resource.ORE] = 19
    view.actions[0]['vertex'] = -1
    assert game.players[0].resources[Resource.ORE] == 0
    assert game.legal_actions(0)[0]['vertex'] >= 0


def test_nested_action_payloads_stay_detached():
    game = Game.new(['a', 'b', 'c', 'd'], 42)
    game.apply(0, game.legal_actions(0)[0])
    view = Observation.from_game(game, 0)
    original = copy.deepcopy(view.actions)
    actions = view.legal_actions(0)
    actions[0]['edge'][0] = -1
    assert view.actions == original
    assert game.legal_actions(0)[0]['edge'][0] >= 0
    from catan_ai.observation import _copy_action
    nested = {'type': 'future', 'payload': [{'cards': [1, 2]}]}
    copied = _copy_action(nested)
    copied['payload'][0]['cards'].append(3)
    assert nested['payload'][0]['cards'] == [1, 2]
