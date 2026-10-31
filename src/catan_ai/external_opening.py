"""External setup search followed by the unchanged economic turn planner."""
from .external_policy import ExternalSearchBot
from .planner import ProductionRobberPlanner


class ExternalOpeningPlanner(ProductionRobberPlanner):
    needs_history = True

    def __init__(self, rng, seed):
        super().__init__(rng)
        self.opening = ExternalSearchBot(seed)
        self.external_diagnostics = self.opening.external_diagnostics

    @property
    def identity(self):
        return self.opening.identity

    def choose(self, view, seat):
        if view.phase.startswith('setup'):
            return self.opening.choose(view, seat)
        if self.opening.process is not None:
            self.opening.close()
        return super().choose(view, seat)
