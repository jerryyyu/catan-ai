from copy import deepcopy
from types import SimpleNamespace

import pytest

from catan_ai.belief import Particle, BeliefFailure
from catan_ai.engine import RESOURCES, Resource
from catan_ai.guided_belief import GuidedBeliefTracker


@pytest.mark.parametrize('kind', ['discard', 'steal'])
def test_guided_proposal_preserves_required_card_and_corrects_weight(kind):
    tracker = GuidedBeliefTracker(0, 1)
    particle = Particle.empty()
    particle.hands[1][Resource.GRAIN] = 8
    particle.hands[1][Resource.ORE] = 1
    event = {'seat': 2 if kind == 'steal' else 1,
             'action': {'type': kind, **({'victim': 1} if kind == 'steal' else {})},
             'counts': [0, 8, 1 if kind == 'steal' else 0, 0],
             'phase': 'discard', 'turn': 1, 'public_points': [0] * 4, 'turn_owner': 0, 'winner': None}
    minimum = [{r: 0 for r in RESOURCES} for _ in range(4)]
    minimum[1][Resource.ORE] = 1
    tracker.minimum_after = {id(event): minimum}
    weight = tracker._advance(particle, event)
    assert weight == pytest.approx(8 / 9)
    assert particle.hands[1][Resource.ORE] == 1
    assert particle.hands[1][Resource.GRAIN] == 7
    if kind == 'steal':
        assert particle.hands[2][Resource.GRAIN] == 1


def test_retrospective_trade_constraint_recovers_with_one_particle():
    base = {'turn': 1, 'phase': 'actions', 'public_points': [0] * 4, 'turn_owner': 0, 'winner': None}
    events = [
        {**base, 'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 9, 1, 0],
         'resource_deltas': [{}, {'grain': 8, 'ore': 1}, {'grain': 1}, {}]},
        {**base, 'seat': 1, 'action': {'type': 'discard'}, 'counts': [0, 8, 1, 0]},
        {**base, 'seat': 1, 'action': {'type': 'accept_trade'}, 'counts': [0, 8, 1, 0],
         'resource_deltas': [{}, {'grain': 1, 'ore': -1}, {'grain': -1, 'ore': 1}, {}]},
    ]
    players = [SimpleNamespace(resources={r: 0 for r in RESOURCES}, resource_count=count,
                               development=(), development_turns=(), development_count=0)
               for count in (0, 8, 1, 0)]
    view = SimpleNamespace(seat=0, players=players, events=events)
    before = deepcopy(events)
    for seed in range(10):
        tracker = GuidedBeliefTracker(seed, 1)
        tracker.update(view)
        assert tracker.population[0].hands[1][Resource.GRAIN] == 8
        assert tracker.population[0].hands[2][Resource.ORE] == 1
    assert events == before
    impossible = deepcopy(view)
    impossible.events[-1]['resource_deltas'][1]['ore'] = -2
    with pytest.raises(BeliefFailure):
        GuidedBeliefTracker(0, 8).update(impossible)


def test_forced_stolen_resource_is_reserved_in_victim_before_earlier_discard():
    base = {'turn': 1, 'phase': 'actions', 'public_points': [0] * 4,
            'turn_owner': 0, 'winner': None}
    events = [
        {**base, 'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 3, 0, 0],
         'resource_deltas': [{}, {'wool': 2, 'grain': 1}, {}, {}]},
        {**base, 'seat': 1, 'action': {'type': 'discard'}, 'counts': [0, 2, 0, 0]},
        {**base, 'seat': 2, 'action': {'type': 'steal', 'victim': 1}, 'counts': [0, 1, 1, 0]},
        {**base, 'seat': 0, 'action': {'type': 'accept_trade'}, 'counts': [2, 0, 0, 0],
         'resource_deltas': [{'wool': 2}, {'wool': -1}, {'wool': -1}, {}]},
    ]
    players = [SimpleNamespace(resources={r: 0 for r in RESOURCES}, resource_count=n,
                               development=(), development_turns=(), development_count=0)
               for n in (2, 0, 0, 0)]
    players[0].resources[Resource.WOOL] = 2
    view = SimpleNamespace(seat=0, players=players, events=events)
    for seed in range(10):
        tracker = GuidedBeliefTracker(seed, 1)
        tracker.update(view)
        assert tracker.minimum_after[id(events[0])][1][Resource.WOOL] == 2
        assert tracker.population[0].hands[0][Resource.WOOL] == 2


def test_one_future_stolen_card_cannot_fill_multiple_resource_shortages():
    base = {'turn': 1, 'phase': 'actions', 'public_points': [0] * 4,
            'turn_owner': 0, 'winner': None}
    events = [
        {**base, 'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 1, 4, 0],
         'resource_deltas': [{}, {'ore': 1}, {'brick': 1, 'grain': 1, 'ore': 1, 'wool': 1}, {}]},
        {**base, 'seat': 2, 'action': {'type': 'discard'}, 'counts': [0, 1, 3, 0]},
        {**base, 'seat': 2, 'action': {'type': 'steal', 'victim': 1}, 'counts': [0, 0, 4, 0]},
        {**base, 'seat': 2, 'action': {'type': 'bank_trade'}, 'counts': [0, 0, 0, 0],
         'resource_deltas': [{}, {}, {'brick': -1, 'grain': -1, 'ore': -2}, {}]},
    ]
    players = [SimpleNamespace(resources={r: 0 for r in RESOURCES}, resource_count=0,
                               development=(), development_turns=(), development_count=0)
               for _ in range(4)]
    view = SimpleNamespace(seat=0, players=players, events=events)
    before = deepcopy(events)
    for seed in range(10):
        tracker = GuidedBeliefTracker(seed, 1)
        tracker.update(view)
        assert tracker.population[0].hands[2] == players[2].resources
    assert events == before


def test_unavailable_recipient_resource_forces_victim_to_keep_it_before_discard():
    base = {'turn': 1, 'phase': 'actions', 'public_points': [0] * 4,
            'turn_owner': 0, 'winner': None}
    events = [
        {**base, 'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 2, 2, 0],
         'resource_deltas': [{}, {'grain': 1, 'wool': 1}, {'wool': 2}, {}]},
        {**base, 'seat': 1, 'action': {'type': 'discard'}, 'counts': [0, 1, 2, 0]},
        {**base, 'seat': 2, 'action': {'type': 'steal', 'victim': 1}, 'counts': [0, 0, 3, 0]},
        {**base, 'seat': 2, 'action': {'type': 'bank_trade'}, 'counts': [0, 0, 0, 0],
         'resource_deltas': [{}, {}, {'grain': -1, 'wool': -2}, {}]},
    ]
    players = [SimpleNamespace(resources={r: 0 for r in RESOURCES}, resource_count=0,
                               development=(), development_turns=(), development_count=0)
               for _ in range(4)]
    view = SimpleNamespace(seat=0, players=players, events=events)
    for seed in range(10):
        tracker = GuidedBeliefTracker(seed, 1)
        tracker.update(view)
        assert tracker.minimum_after[id(events[0])][1][Resource.GRAIN] == 1
        assert tracker.population[0].hands[2] == players[2].resources


def test_group_conservation_guides_discard_when_recipient_hand_is_uncertain():
    base = {'turn': 1, 'phase': 'actions', 'public_points': [0] * 4,
            'turn_owner': 0, 'winner': None}
    events = [
        {**base, 'seat': 0, 'action': {'type': 'roll'}, 'counts': [0, 2, 0, 2],
         'resource_deltas': [{}, {'grain': 1, 'wool': 1}, {}, {'grain': 1, 'wool': 1}]},
        {**base, 'seat': 2, 'action': {'type': 'steal', 'victim': 3}, 'counts': [0, 2, 1, 1]},
        {**base, 'seat': 1, 'action': {'type': 'discard'}, 'counts': [0, 1, 1, 1]},
        {**base, 'seat': 2, 'action': {'type': 'steal', 'victim': 1}, 'counts': [0, 0, 2, 1]},
        {**base, 'seat': 2, 'action': {'type': 'bank_trade'}, 'counts': [0, 0, 0, 1],
         'resource_deltas': [{}, {}, {'grain': -1, 'wool': -1}, {}]},
    ]
    players = [SimpleNamespace(resources={r: 0 for r in RESOURCES}, resource_count=n,
                               development=(), development_turns=(), development_count=0)
               for n in (0, 0, 0, 1)]
    view = SimpleNamespace(seat=0, players=players, events=events)
    for seed in range(20):
        tracker = GuidedBeliefTracker(seed, 1)
        tracker.update(view)
        assert tracker.population[0].hands[2] == players[2].resources
        assert tracker.group_after[id(events[2])]
