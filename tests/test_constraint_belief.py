from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip('z3', reason='run with --group solver for constraint inference')

from catan_ai.belief import BeliefFailure
from catan_ai.constraint_belief import solve_development, solve_particle, solve_resources, verify_witness
from catan_ai.engine import RESOURCES


def view_for(events, seat=0, own_cards=(), counts=(0, 0, 0, 0)):
    players = [SimpleNamespace(resources={r: 0 for r in RESOURCES}, resource_count=0,
                               development=tuple(c for c, _ in own_cards) if s == seat else None,
                               development_turns=tuple(t for _, t in own_cards) if s == seat else None,
                               development_count=counts[s]) for s in range(4)]
    return SimpleNamespace(events=events, seat=seat, players=players)


def event(action, seat=1, turn=0, points=0, winner=None):
    return {'action': action, 'seat': seat, 'turn': turn, 'turn_owner': seat,
            'phase': 'actions', 'public_points': [points] * 4, 'winner': winner,
            'counts': [0] * 4, 'resource_deltas': [{} for _ in range(4)]}


@pytest.mark.parametrize('resource', RESOURCES)
def test_hidden_resource_choice_moves_exactly_one_available_card(resource):
    grant = event({'type': 'setup_settlement'}, seat=1)
    grant['resource_deltas'][1] = {resource.value: 1}
    grant['counts'] = [0, 1, 0, 0]
    steal = event({'type': 'steal', 'victim': 1}, seat=2)
    del steal['resource_deltas']
    steal['counts'] = [0, 0, 1, 0]
    view = view_for([grant, steal])
    assert solve_resources(view) == {1: resource.value}
    steal['counts'] = [0, 0, 2, 0]
    with pytest.raises(RuntimeError, match='unsat'):
        solve_resources(view)


def test_development_age_and_played_identity_are_required():
    events = [event({'type': 'buy_development'}),
              event({'type': 'play_development', 'card': 'monopoly'})]
    with pytest.raises(BeliefFailure):
        solve_development(view_for(events))
    events[1]['turn'] = 1
    assert solve_development(view_for(events)) == {0: 'monopoly'}


def test_deck_supply_and_observed_draws_cannot_be_changed():
    events = [{**event({'type': 'buy_development'}), 'drawn_card': 'monopoly'} for _ in range(3)]
    with pytest.raises(BeliefFailure):
        solve_development(view_for(events, counts=(0, 3, 0, 0)))
    events = [{**event({'type': 'buy_development'}, seat=0), 'drawn_card': 'knight'}]
    view = view_for(events, own_cards=(('knight', 0),), counts=(1, 0, 0, 0))
    with pytest.raises(BeliefFailure, match='observed development'):
        verify_witness(view, {}, {0: 'monopoly'})
    before = deepcopy(view)
    assert solve_particle(view).development[0] == [('knight', 0)]
    assert view == before


def test_hidden_victory_cards_respect_observed_winner():
    events = [event({'type': 'buy_development'}, points=9)]
    assert solve_development(view_for(events, counts=(0, 1, 0, 0)))[0] != 'victory_point'
    events[0]['winner'] = 1
    assert solve_development(view_for(events, counts=(0, 1, 0, 0)))[0] == 'victory_point'


def test_verifier_rejects_missing_or_extra_choices():
    view = view_for([event({'type': 'end_turn'})])
    with pytest.raises(BeliefFailure, match='choices'):
        verify_witness(view, {0: 'grain'}, {})
    view.events[0]['action'] = {'type': 'buy_development'}
    with pytest.raises(BeliefFailure, match='choices'):
        verify_witness(view, {}, {})
