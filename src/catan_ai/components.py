"""Controlled opening/turn policy combinations for component experiments."""
import random

from .external_opening import ExternalOpeningPlanner
from .tactical import TacticalBot


class ComponentPolicy:
    needs_history = False

    def __init__(self, opening, turn, seed):
        from .bots import make_policy
        self.opening = make_policy(opening, seed)
        self.turn = make_policy(turn, seed)

    def choose(self, view, seat):
        policy = self.opening if view.phase.startswith('setup') else self.turn
        return policy.choose(view, seat)


class ExternalOpeningTactical(ExternalOpeningPlanner):
    def __init__(self, rng, seed):
        super().__init__(rng, seed)
        self.turn_policy = TacticalBot(random.Random(seed))

    def choose(self, view, seat):
        if view.phase.startswith('setup'):
            return super().choose(view, seat)
        if self.opening.process is not None:
            self.opening.close()
        return self.turn_policy.choose(view, seat)
