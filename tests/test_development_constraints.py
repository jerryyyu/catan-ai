from copy import deepcopy

import pytest

from catan_ai.belief import Particle
from catan_ai.development_constraints import draw_constraints, feasible_draw
from catan_ai.engine import RESOURCES
from catan_ai.guided_belief import GuidedBeliefTracker


def event(kind, turn, seat=1, **action):
    return {'seat': seat, 'turn': turn, 'phase': 'actions',
            'action': {'type': kind, **action}, 'counts': [0] * 4,
            'resource_deltas': [{} for _ in range(4)], 'public_points': [0] * 4,
            'turn_owner': seat, 'winner': None}


def test_forced_hidden_draw_keeps_prior_mass_and_deck_conservation():
    buy = event('buy_development', 1)
    play = event('play_development', 2, card='year_of_plenty')
    tracker = GuidedBeliefTracker(0, 1)
    tracker.draw_requirements = draw_constraints([buy, play])
    tracker.minimum_after = {id(buy): [{r: 0 for r in RESOURCES} for _ in range(4)]}
    particle = Particle.empty()
    before = deepcopy(buy)
    weight = tracker._advance(particle, buy)
    assert weight == pytest.approx(2 / 25)
    assert particle.development[1] == [('year_of_plenty', 1)]
    assert particle.deck.count('year_of_plenty') == 1
    assert len(particle.deck) == 24
    assert buy == before


def test_same_turn_purchase_cannot_supply_an_observed_play():
    buy = event('buy_development', 1)
    later_buy = event('buy_development', 2)
    play = event('play_development', 2, card='monopoly')
    constraints = draw_constraints([buy, later_buy, play])[id(buy)]
    particle = Particle.empty()
    assert feasible_draw(particle, 1, 'monopoly', 1, constraints)
    assert not feasible_draw(particle, 1, 'knight', 1, constraints)


def test_eligible_later_draw_does_not_force_earlier_identity():
    buy = event('buy_development', 1)
    later_buy = event('buy_development', 2)
    play = event('play_development', 3, card='monopoly')
    constraints = draw_constraints([buy, later_buy, play])[id(buy)]
    assert feasible_draw(Particle.empty(), 1, 'knight', 1, constraints)


def test_other_players_known_draw_reserves_last_deck_copy():
    buy = event('buy_development', 1)
    known = {**event('buy_development', 2, seat=2), 'drawn_card': 'monopoly'}
    particle = Particle.empty()
    particle.deck.remove('monopoly')
    particle.development[3].append(('monopoly', 0))
    constraints = draw_constraints([buy, known])[id(buy)]
    assert not feasible_draw(particle, 1, 'monopoly', 1, constraints)
    assert feasible_draw(particle, 1, 'knight', 1, constraints)
