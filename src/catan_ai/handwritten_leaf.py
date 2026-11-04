"""Hand-written build-search leaf: every constant written from game knowledge, none fitted.

``planner_handwritten_leaf`` is the served ``planner_offer_accept`` whose turn-search
leaf is the function below; every other decision is unchanged. It follows the
weight-space-programming idea (an LLM writing a model's parameters directly from
the rules, with no data) and was written in one pass on 2026-11-03, after the
learned leaves failed DEV, with these design goals taken from those failures:

- building must never look worse than holding the cards (no purchase-deficit
  terms; a card in hand is worth far less than what it builds);
- a 4:1 bank trade costs three cards' value unless it completes a build;
- a hand over seven carries the expected cost of a 7;
- the scale keeps one victory point at 12 units, the served leaf's scale, so the
  search's fixed development-purchase bonus keeps its intended weight.

Units: one victory point = 12.
"""
import math

from .engine import Resource
from .unified_planner import AcceptanceOfferPlanner

VP = 12.0                 # per victory point, public or hidden
PIP = {Resource.BRICK: 1.0, Resource.LUMBER: 1.0, Resource.WOOL: .8,
       Resource.GRAIN: 1.1, Resource.ORE: 1.1}   # per production pip (dots on the number)
DIVERSITY = 1.5           # extra value of sqrt(pips) per resource: covering a resource matters most
CARD = .8                 # per resource card in hand
OVER_SEVEN = 1.5          # per card above seven (expected loss to a 7 before the next turn)
KNIGHT = 3.0              # per knight played (Largest Army race)
DEV_CARD = 2.0            # per unplayed non-VP development card
ROAD = .5                 # per road (reach, Longest Road)
FIRST_SPOT = 4.0          # having at least one legal settlement spot
SECOND_SPOT = 1.0         # having a second one
GENERIC_PORT = 2.0        # a 3:1 port
SPECIFIC_PORT = 1.5       # each 2:1 port
OPPONENT = .5             # weight on the strongest opponent's value (served shared-score lambda)
WIN = 1e5


def hand_value(game, seat, income):
    player = game.players[seat]
    points = player.public_points + sum(card == 'victory_point' for card, _ in player.development)
    if points >= 10:
        return WIN
    value = VP * points
    value += sum(PIP[r] * n + DIVERSITY * math.sqrt(max(n, 0.)) for r, n in income.items())
    cards = sum(player.resources.values())
    value += CARD * cards - OVER_SEVEN * max(0, cards - 7)
    value += KNIGHT * player.knights
    value += DEV_CARD * sum(card != 'victory_point' for card, _ in player.development)
    value += ROAD * len(player.roads)
    spots = len(game._legal_settlements(seat))
    value += FIRST_SPOT * (spots >= 1) + SECOND_SPOT * (spots >= 2)
    rates = [game.trade_rate(seat, r) for r in Resource]
    value += GENERIC_PORT * (3 in rates)  # a 3:1 port sets every non-2:1 rate to 3
    value += SPECIFIC_PORT * sum(rate == 2 for rate in rates)
    return value


class HandWrittenLeafPlanner(AcceptanceOfferPlanner):
    """``planner_offer_accept`` with the hand-written leaf above in its build search."""

    def _leaf(self, game, seat):
        mine = hand_value(game, seat, self._income(game, seat))
        if mine >= WIN:
            return WIN
        threat = max(hand_value(game, s, self._income(game, s)) for s in range(len(game.players)) if s != seat)
        return mine - OPPONENT * min(threat, WIN / 10)
