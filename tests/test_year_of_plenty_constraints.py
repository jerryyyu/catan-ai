from copy import deepcopy
from dataclasses import replace
from itertools import product

import pytest

z3 = pytest.importorskip('z3')

from catan_ai.constraint_belief import constrain_year_of_plenty, solve_resources
from catan_ai.engine import Game, RESOURCES, Resource
from catan_ai.observation import Observation


def position(bank, phase='roll', bought=0, played=False):
    game = Game.new(['a', 'b', 'c', 'd'], 1621000)
    game.phase, game.turn, game.current = phase, 2, 0
    game.played_development = played
    game.players[0].development = [('year_of_plenty', bought)]
    for resource, available in zip(RESOURCES, bank):
        game.players[1].resources[resource] = 19 - available
    return game


def card_actions(game):
    return [a for a in game.legal_actions(0)
            if a['type'] == 'play_development' and a['card'] == 'year_of_plenty']


@pytest.mark.parametrize('phase', ['roll', 'actions'])
def test_bank_constraints_agree_with_engine_for_every_zero_one_two_inventory(phase):
    variables = z3.Ints('brick lumber wool grain ore')
    for bank in product(range(3), repeat=5):
        game = position(bank, phase)
        view = Observation.from_game(game, 0)
        before = deepcopy(view)
        assert all(p.resources is None for p in view.players[1:])
        solver = z3.Solver()
        constrain_year_of_plenty(solver, view, variables)
        solver.push()
        solver.add(*(x == n for x, n in zip(variables, bank)))
        assert solver.check() == z3.sat, bank
        solver.pop()
        alternative = ((bank[0] + 1) % 3,) + bank[1:]
        solver.add(*(x == n for x, n in zip(variables, alternative)))
        expected = card_actions(game) == card_actions(position(alternative, phase))
        assert (solver.check() == z3.sat) == expected, (bank, alternative)
        assert view == before


@pytest.mark.parametrize('phase,bought,played', [
    ('roll', 2, False), ('actions', 0, True), ('free_roads', 0, False),
])
def test_ineligible_card_adds_no_bank_constraints(phase, bought, played):
    view = Observation.from_game(position((1, 0, 0, 0, 0), phase, bought, played), 0)
    solver = z3.Solver()
    constrain_year_of_plenty(solver, view, z3.Ints('b l w g o'))
    assert len(solver.assertions()) == 0


@pytest.mark.parametrize('phase', ['roll', 'actions'])
@pytest.mark.parametrize('seed', range(8))
def test_observed_card_choice_constrains_hidden_discard_in_resource_solver(phase, seed):
    game = position((19, 19, 1, 19, 18), phase)
    assert RESOURCES[2] == Resource.WOOL
    view = Observation.from_game(game, 0)
    grant = {'seat': 1, 'action': {'type': 'setup_settlement'}, 'counts': [0, 20, 0, 0],
             'resource_deltas': [{}, {'wool': 19, 'ore': 1}, {}, {}]}
    discard = {'seat': 1, 'action': {'type': 'discard'}, 'counts': [0, 19, 0, 0]}
    view = replace(view, events=(grant, discard))
    before = deepcopy(view)
    assert solve_resources(view, seed) == {1: 'wool'}
    assert view == before
