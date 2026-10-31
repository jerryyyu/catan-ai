"""Delay economic development cards until the turn's production is known."""
from .planner import ProductionRobberPlanner


class CardTimingPlanner(ProductionRobberPlanner):
    def choose(self, view, seat):
        if view.phase == 'roll':
            knight = next((a for a in view.actions if a['type'] == 'play_development'
                           and a['card'] == 'knight'), None)
            if knight is not None:
                return knight
            return next(a for a in view.actions if a['type'] == 'roll')
        return super().choose(view, seat)
