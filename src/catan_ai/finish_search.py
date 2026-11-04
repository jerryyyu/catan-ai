"""Board row 5: next-turn finish search for the served bot.

``planner_finish_next`` is ``planner_offer_accept`` that, once its own points
(including hidden victory-point cards) reach ``FINISH_POINTS``, scores each
candidate first action by the chance of winning on its next turn, and overrides
the served choice only when that chance is clearly higher.

For a candidate, the search follows short continuations of builds, bank trades
and development purchases (``FINISH_DEPTH`` actions, ``FINISH_NODES`` nodes per candidate),
then ends the turn. Each end-of-turn state is scored by ``FINISH_SAMPLES``
fixed samples of the four dice rolls before its next turn (common random numbers
across candidates): its own production per roll (the robber's tile produces
nothing), a 7 halving a hand over seven cards (largest piles first), then
whether the resulting hand, with bank trades at its own rates, can pay for the
cities and settlements (on currently legal spots) that reach 10 points. Awards,
development cards drawn later, opponents' builds, robber moves and opponents
winning first are ignored: this is a bounded own-finish estimate, not a game
solver. A development purchase counts as a victory point with probability
``VP_SHARE``. Uses public state, its own hand and its own cards only.
"""
from copy import deepcopy
import random

from .engine import COSTS, Resource
from .unified_planner import AcceptanceOfferPlanner

# Predeclared 2026-11-03 before any game (board row 5).
FINISH_POINTS = 8     # activate at this many own points (hidden VP cards included)
FINISH_DEPTH = 3      # actions per candidate continuation before the turn ends
FINISH_NODES = 32     # search nodes per candidate first action
FINISH_SAMPLES = 128  # dice samples for the next round (four rolls)
FINISH_MARGIN = .05   # override the served move only for this much more win chance
VP_SHARE = 5 / 25     # victory-point share of the standard development deck
_RES = tuple(Resource)
_KINDS = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development')


def _roll(rng):
    return rng.randint(1, 6) + rng.randint(1, 6)


def _production(game, seat):
    table = {n: [0] * 5 for n in range(2, 13)}
    player = game.players[seat]
    for i, tile in enumerate(game.board.tiles):
        if i == game.robber_tile or tile.number is None or not isinstance(tile.resource, Resource):
            continue
        k = _RES.index(tile.resource)
        for v in tile.vertices:
            table[tile.number][k] += 2 if v in player.cities else 1 if v in player.settlements else 0
    return table


def _affordable(hand, cost, rates):
    deficit = sum(max(0, cost[k] - hand[k]) for k in range(5))
    surplus = sum(max(0, hand[k] - cost[k]) // rates[k] for k in range(5))
    return surplus >= deficit


def _can_finish(hand, rates, needed, spots, settlements, cities_left, settlements_left):
    city = [COSTS['city'].get(r, 0) for r in _RES]
    settle = [COSTS['settlement'].get(r, 0) for r in _RES]
    for m in range(0, min(spots, settlements_left, needed) + 1):
        k = needed - m
        if k > min(cities_left, settlements + m):
            continue
        cost = [k * c + m * s for c, s in zip(city, settle)]
        if _affordable(hand, cost, rates):
            return True
    return False


class NextTurnFinishPlanner(AcceptanceOfferPlanner):
    def choose(self, view, seat):
        served = super().choose(view, seat)
        player = view.players[seat]
        own_points = player.public_points + (player.development or ()).count('victory_point')
        if (view.phase != 'actions' or own_points < FINISH_POINTS
                or served['type'] not in _KINDS + ('end_turn',)):
            return served
        root = self._world(view)
        rng = random.Random(1_000_003 * view.turn + seat)
        rolls = [[_roll(rng) for _ in range(4)] for _ in range(FINISH_SAMPLES)]
        cache, nodes = {}, [0]

        def chance(game, vp_bonus=0):
            p = game.players[seat]
            needed = 10 - p.points - vp_bonus
            if needed <= 0:
                return 1.
            key = (tuple(p.resources[r] for r in _RES), frozenset(p.settlements), frozenset(p.cities),
                   frozenset(p.roads), needed)
            if key in cache:
                return cache[key]
            table = _production(game, seat)
            rates = [game.trade_rate(seat, r) for r in _RES]
            spots = len(game._legal_settlements(seat))
            settlements, cities_left = len(p.settlements), 4 - len(p.cities)
            settlements_left = 5 - len(p.settlements)
            wins = 0
            for sample in rolls:
                hand = [p.resources[r] for r in _RES]
                for n in sample:
                    if n == 7:
                        total = sum(hand)
                        for _ in range(total // 2 if total > 7 else 0):
                            hand[hand.index(max(hand))] -= 1
                    else:
                        hand = [a + b for a, b in zip(hand, table[n])]
                wins += _can_finish(hand, rates, needed, spots, settlements, cities_left, settlements_left)
            cache[key] = wins / len(rolls)
            return cache[key]

        def best(game, depth):
            value = chance(game)
            if depth == 0:
                return value
            for action in game.legal_actions(seat):
                if action['type'] not in _KINDS or nodes[0] >= FINISH_NODES:
                    continue
                nodes[0] += 1
                child = deepcopy(game, {id(game.board): game.board})
                if action['type'] == 'buy_development':
                    child.players[seat].pay('development')
                    value = max(value, VP_SHARE * chance(child, 1) + (1 - VP_SHARE) * best(child, depth - 1))
                    continue
                child.apply(seat, action)
                if child.winner == seat:
                    return 1.
                value = max(value, best(child, depth - 1))
            return value

        def first_value(action):
            nodes[0] = 0
            if action['type'] == 'end_turn':
                return chance(root)
            child = deepcopy(root, {id(root.board): root.board})
            if action['type'] == 'buy_development':
                child.players[seat].pay('development')
                return VP_SHARE * chance(child, 1) + (1 - VP_SHARE) * best(child, FINISH_DEPTH - 1)
            child.apply(seat, action)
            return 1. if child.winner == seat else best(child, FINISH_DEPTH - 1)

        # Candidates come from the card-counted world, whose bank stock can differ from
        # the real one; a served move that is not legal there is kept as is.
        legal = root.legal_actions(seat)
        if served not in legal:
            return served
        candidates = [a for a in legal if a['type'] in _KINDS + ('end_turn',) and a in view.actions]
        served_value = first_value(served)
        scored = [(first_value(a), i, a) for i, a in enumerate(candidates)]
        value, _, action = max(scored, key=lambda t: (t[0], -t[1]))
        return action if value > served_value + FINISH_MARGIN else served
