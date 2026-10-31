import pytest

from catan_ai.belief import BeliefFailure, Particle
from catan_ai.engine import Game, Resource
from catan_ai.guided_belief import GuidedBeliefTracker
from catan_ai.observation import Observation
from catan_ai.recovery_belief import RecoveryBeliefTracker


def test_particle_exhaustion_recovers_once_with_independent_copies(monkeypatch):
    view = Observation.from_game(Game.new(['a', 'b', 'c', 'd'], 1453000), 0)
    calls = []

    def failed_update(self, observation):
        raise BeliefFailure('forced exhaustion')

    def solve(observation, seed, timeout_ms):
        assert observation is view
        assert timeout_ms == 500
        calls.append(seed)
        return Particle.empty()

    monkeypatch.setattr(GuidedBeliefTracker, 'update', failed_update)
    monkeypatch.setattr('catan_ai.recovery_belief.solve_particle', solve)
    tracker = RecoveryBeliefTracker(7, 4)
    world = tracker.sample(view, 1)
    assert world.legal_actions(0) == list(view.actions)
    assert calls == [7]
    assert tracker.solver_attempts == tracker.solver_recoveries == 1
    tracker.population[0].hands[0][Resource.GRAIN] += 1
    assert all(p.hands[0][Resource.GRAIN] == 0 for p in tracker.population[1:])


def test_failed_recovery_cannot_retain_stale_particles_or_retry(monkeypatch):
    view = Observation.from_game(Game.new(['a', 'b', 'c', 'd'], 1453001), 0)

    def failed(*args, **kwargs):
        raise BeliefFailure('unsatisfiable')

    monkeypatch.setattr(GuidedBeliefTracker, 'update', failed)
    monkeypatch.setattr('catan_ai.recovery_belief.solve_particle', failed)
    tracker = RecoveryBeliefTracker(7, 4)
    tracker.population = [Particle.empty()]
    with pytest.raises(BeliefFailure, match='unsatisfiable'):
        tracker.sample(view, 1)
    assert tracker.population == []
    assert tracker.solver_attempts == 1 and tracker.solver_recoveries == 0


def test_existing_guided_tracker_keeps_original_replay_budget():
    assert GuidedBeliefTracker.replay_attempts == 3
    assert RecoveryBeliefTracker.replay_attempts == 1
