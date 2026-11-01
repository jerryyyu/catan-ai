"""Candidates screened against the served fitted planner.

Each keeps the fitted weights of ``planner_parameter`` and changes one
behaviour, so a screen against the served planner isolates that change.

``planner_discard`` (row 9) prices the risk of losing half of a hand over seven
cards to a 7 before the next turn, instead of a flat 0.5 per card, and chooses
which card to discard by the same leaf value instead of "largest pile".

``planner_devfix`` fixes the served planner's random development-card play
before the roll: the inherited rules played a random card in the roll phase, so
every card was spent before the turn search could use Year of Plenty or Road
Building. It rolls instead. Development cards are still played by the inherited
rules after the search; only the pre-roll play is removed.
"""
from .engine import Resource
from .parameter_planner import ParameterPlanner
from .planner import WEIGHTS
from .tactical import public_projection

# Probability that at least one of the three opponents' rolls is a 7 before our
# next turn. The hand left at the end of our turn is exposed to those rolls.
P_SEVEN_BEFORE_NEXT_TURN = 1 - (5 / 6) ** 3


class DiscardAwarePlanner(ParameterPlanner):
    def _hand_value(self, player):
        return sum(WEIGHTS[r] * (1.6 * min(n, 3) + .45 * max(n - 3, 0)) for r, n in player.resources.items())

    def _value(self, game, seat):
        value = super()._value(game, seat)
        player = game.players[seat]
        if player.points >= 10:
            return value
        cards = sum(player.resources.values())
        flat = .5 * max(cards - 7, 0)
        expected = 0.0
        if cards > 7:
            per_card = self.weights[2] * self._hand_value(player) / cards
            expected = P_SEVEN_BEFORE_NEXT_TURN * (cards // 2) * per_card
        return value + flat - expected

    def choose(self, view, seat):
        discards = [a for a in view.actions if a['type'] == 'discard']
        if view.phase != 'discard' or not discards:
            return super().choose(view, seat)
        root = public_projection(view)
        player = root.players[seat]

        def kept_value(action):
            resource = Resource(action['resource'])
            player.resources[resource] -= 1
            try:
                return self._value(root, seat)
            finally:
                player.resources[resource] += 1
        # max keeps the first of equal values, so ties resolve in legal-action order.
        return max(discards, key=kept_value)


class DevelopmentFixPlanner(ParameterPlanner):
    def _fallback(self, view, seat):
        if view.phase == 'roll':
            return next(a for a in view.actions if a['type'] == 'roll')
        return super()._fallback(view, seat)
