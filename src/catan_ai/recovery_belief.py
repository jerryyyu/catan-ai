"""Incremental particle tracking with verified, non-posterior solver recovery."""
from copy import deepcopy
import time

from .belief import BeliefFailure
from .constraint_belief import solve_particle
from .guided_belief import GuidedBeliefTracker


class RecoveryBeliefTracker(GuidedBeliefTracker):
    replay_attempts = 1

    def __init__(self, seed, particles=512):
        super().__init__(seed, particles)
        self.solver_seed = seed
        self.solver_attempts = 0
        self.solver_recoveries = 0
        self.solver_seconds = 0.0

    def _recover(self, view):
        self.population = []
        self.solver_attempts += 1
        started = time.perf_counter()
        try:
            particle = solve_particle(view, self.solver_seed, timeout_ms=500)
            self.population = [deepcopy(particle) for _ in range(self.size)]
            self.history, self.seat = deepcopy(list(view.events)), view.seat
            self.solver_recoveries += 1
        finally:
            self.solver_seconds += time.perf_counter() - started

    def update(self, view):
        prefix = (self.seat == view.seat and len(view.events) >= len(self.history)
                  and list(view.events[:len(self.history)]) == self.history and self.population)
        if not prefix and len(view.events) > 32:
            self._recover(view)
            return
        try:
            super().update(view)
        except BeliefFailure:
            self._recover(view)

    def sample(self, view, seed):
        attempts = self.solver_attempts
        try:
            return super().sample(view, seed)
        except BeliefFailure:
            if self.solver_attempts != attempts:
                raise
            self._recover(view)
            return super().sample(view, seed)
