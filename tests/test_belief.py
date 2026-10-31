from collections import Counter
import copy

import pytest

from catan_ai.belief import BeliefFailure, BeliefTracker, Particle
from catan_ai.bots import make_policy
from catan_ai.engine import Game, RESOURCES
from catan_ai.observation import Observation


def test_sampled_worlds_reproduce_information_and_conserve_cards():
    game = Game.new(['a', 'b', 'c', 'd'], 29)
    bots = [make_policy('heuristic', s) for s in range(4)]
    trackers = [BeliefTracker(s, 64) for s in range(4)]
    for step in range(90):
        seat = game.current
        view = Observation.from_game(game, seat)
        sampled = trackers[seat].sample(view, step)
        assert sampled.legal_actions(seat) == list(view.actions)
        assert sampled.players[seat].resources == game.players[seat].resources
        assert sampled.players[seat].development == game.players[seat].development
        assert all(0 <= sampled.bank[r] <= 19 for r in RESOURCES)
        assert [sum(p.resources.values()) for p in sampled.players] == [p.resource_count for p in view.players]
        cards = Counter(sampled.deck)
        cards.update(c for p in sampled.players for c, _ in p.development)
        cards.update(e['action']['card'] for e in view.events if e['action']['type'] == 'play_development')
        assert cards == Counter({'knight': 14, 'victory_point': 5, 'monopoly': 2,
                                 'year_of_plenty': 2, 'road_building': 2})
        assert sampled.events == []
        game.apply(seat, bots[seat].choose(view, seat))


def test_sampling_uses_no_authoritative_hidden_state_or_rng():
    game = Game.new(['a', 'b', 'c', 'd'], 29)
    changed = copy.deepcopy(game)
    changed.deck.reverse()
    changed.rng.seed(999)
    a, b = Observation.from_game(game, 0), Observation.from_game(changed, 0)
    x = BeliefTracker(14).sample(a, 7)
    y = BeliefTracker(14).sample(b, 7)
    assert x.public_state(0) == y.public_state(0)
    assert x.deck == y.deck
    assert x.rng.getstate() == y.rng.getstate()


def test_missing_history_fails_explicitly_instead_of_using_true_hand():
    game = Game.new(['a', 'b', 'c', 'd'], 29)
    game.players[0].resources[RESOURCES[0]] = 1
    with pytest.raises(BeliefFailure):
        BeliefTracker(0).sample(Observation.from_game(game, 0), 0)


def test_public_shortage_observation_rejects_world_with_available_supply():
    tracker = BeliefTracker(0)
    particle = Particle.empty()
    claims = [{r.value: 0 for r in RESOURCES} for _ in range(4)]
    claims[0]['ore'] = 1
    claims[1]['ore'] = 1
    event = {'seat': 0, 'turn': 0, 'phase': 'roll', 'action': {'type': 'roll'},
             'counts': [0, 0, 0, 0], 'resource_deltas': [{}, {}, {}, {}],
             'production_claims': claims, 'public_points': [2] * 4,
             'winner': None, 'turn_owner': 0}
    assert tracker._advance(particle, event) == 0


def test_unannounced_victory_and_unaffordable_offer_are_impossible():
    tracker = BeliefTracker(0)
    particle = Particle.empty()
    particle.development[0] = [('victory_point', 0)] * 2
    event = {'seat': 0, 'turn': 4, 'phase': 'actions', 'action': {'type': 'bank_trade'},
             'counts': [0] * 4, 'resource_deltas': [{}, {}, {}, {}],
             'public_points': [8, 2, 2, 2], 'winner': None, 'turn_owner': 0}
    assert tracker._advance(particle, event) == 0
    event['action'] = {'type': 'offer_trade', 'recipient': 1, 'give': {'ore': 1}, 'receive': {'wool': 1}}
    assert tracker._advance(Particle.empty(), event) == 0


def test_equal_likelihood_private_draw_does_not_resample_distinct_hands(monkeypatch):
    from catan_ai.engine import Resource
    game = Game.new(['a', 'b', 'c', 'd'], 29)
    game.phase, game.turn = 'actions', 1
    game.deck[-1] = 'knight'
    for r in (Resource.ORE, Resource.GRAIN, Resource.WOOL):
        game.players[0].resources[r] = 1
    game.players[1].resources[Resource.BRICK] = 1
    tracker = BeliefTracker(0, particles=2)
    tracker.seat = 0
    for hidden in (Resource.BRICK, Resource.LUMBER):
        particle = Particle.empty()
        particle.hands[0] = dict(game.players[0].resources)
        particle.hands[1][hidden] = 1
        tracker.population.append(particle)
    game.apply(0, {'type': 'buy_development'})
    def unexpected_resampling(*args, **kwargs):
        raise AssertionError('equal weights do not change relative probability')
    monkeypatch.setattr(tracker.rng, 'choices', unexpected_resampling)
    tracker.update(Observation.from_game(game, 0))
    assert tracker.population[0].hands[1] != tracker.population[1].hands[1]
