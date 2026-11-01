"""One shared position score for every decision outside the build search.

``planner_unified`` keeps the served planner's fitted weights and its own-turn
build search unchanged. Everything that today falls through to fixed
HeuristicBot rules (robber, steal, discard, development-card play, answering
trade offers) instead picks the legal move whose resulting position scores
highest under one function::

    S = V(me) - LAMBDA * max over opponents V(opponent)

``V`` is the fitted leaf value. For hand-shaping decisions (answering trades,
discarding) my side is the best value my next turn's builds reach from the
resulting hand, so cards held for an affordable build keep their worth. Opponents' hands are public card-counting
estimates (``card_planner.resource_estimates``). Unknown development cards are
scored as non-VP cards. Two scoring changes apply only inside ``S``: production
on the robber's tile is not counted, and knights already played keep the value
they had in hand, so a Knight is played for its board effect and Largest Army
rather than penalised.

The planner never offers trades. Before rolling it plays only a Knight, and
only when the robber blocks one of its own producing tiles and playing scores
higher than holding.
"""
from copy import deepcopy

from .card_planner import resource_estimates
from .engine import Resource, award_holder, longest_road
from .parameter_planner import ParameterPlanner
from .tactical import public_projection

LAMBDA = 0.5
_RESOURCES = list(Resource)


class UnifiedPlanner(ParameterPlanner):
    needs_history = True
    _scoring = False

    # ------------------------------------------------------------ scoring
    def _income(self, game, seat):
        income = super()._income(game, seat)
        if not self._scoring:
            return income
        tile = game.board.tiles[game.robber_tile]
        if tile.number is not None and isinstance(tile.resource, Resource):
            player = game.players[seat]
            blocked = sum((v in player.settlements) + 2 * (v in player.cities) for v in tile.vertices)
            income[tile.resource] -= blocked * (6 - abs(7 - tile.number))
        return income

    def _seat_value(self, game, seat):
        player = game.players[seat]
        return self._value(game, seat) + self.weights[5] * 3 * player.knights

    def _score(self, game, seat):
        self._scoring = True
        try:
            mine = self._seat_value(game, seat)
            threat = max(self._seat_value(game, s) for s in range(len(game.players)) if s != seat)
        finally:
            self._scoring = False
        return mine - LAMBDA * threat

    def _world(self, view):
        """Public projection with opponents' estimated hands and the real robber."""
        game = public_projection(view)
        game.robber_tile = view.robber_tile
        estimates = resource_estimates(view.events)
        for s, player in enumerate(game.players):
            if s == view.seat:
                continue
            player.resources = {r: estimates[s][r] for r in _RESOURCES}
            player.development = [('knight', -1)] * view.players[s].development_count
        return game

    def _plan_value(self, game, seat):
        """My best fitted value over at most two builds or a card purchase, as on my next turn.

        Used for hand-shaping decisions (trade answers, discards) so that keeping
        the cards for an affordable build counts, not just their resource weights.
        """
        start = self._copy(game)
        start.phase, start.current, start.rolled = 'actions', seat, 6
        best = self._seat_value(start, seat)
        frontier = [start]
        for depth in range(2):
            successors = []
            for state in frontier:
                for action in state.legal_actions(seat):
                    kind = action['type']
                    if kind not in ('build_city', 'build_settlement', 'build_road', 'buy_development'):
                        continue
                    child = self._copy(state)
                    if kind == 'buy_development':
                        child.players[seat].pay('development')
                        best = max(best, self._seat_value(child, seat) + self.purchase_value)
                        continue
                    child.apply(seat, action)
                    best = max(best, self._seat_value(child, seat))
                    if depth == 0 and kind != 'build_road':
                        successors.append(child)
            frontier = successors
        return best

    def _hand_score(self, game, seat):
        self._scoring = True
        try:
            mine = self._plan_value(game, seat)
            threat = max(self._seat_value(game, s) for s in range(len(game.players)) if s != seat)
        finally:
            self._scoring = False
        return mine - LAMBDA * threat

    @staticmethod
    def _copy(game):
        return deepcopy(game, {id(game.board): game.board})

    # --------------------------------------------------- move evaluation
    def _steal_value(self, game, seat, victim):
        hand = game.players[victim].resources
        total = sum(hand.values())
        if total <= 0:
            return self._score(game, seat)
        expected = 0.0
        for r in _RESOURCES:
            if hand[r] <= 0:
                continue
            child = self._copy(game)
            child.players[victim].resources[r] -= 1
            child.players[seat].resources[r] += 1
            expected += hand[r] / total * self._score(child, seat)
        return expected

    def _victims(self, game, seat, tile_index):
        occupied = game.occupied()
        return sorted({occupied[v] for v in game.board.tiles[tile_index].vertices
                       if v in occupied and occupied[v] != seat
                       and sum(game.players[occupied[v]].resources.values()) > 0})

    def _robber_value(self, game, seat, tile_index):
        child = self._copy(game)
        child.robber_tile = tile_index
        victims = self._victims(child, seat, tile_index)
        if not victims:
            return self._score(child, seat)
        return max(self._steal_value(child, seat, v) for v in victims)

    def _best_robber(self, game, seat, tiles):
        return max(tiles, key=lambda t: self._robber_value(game, seat, t))

    def _knight_value(self, game, seat):
        child = self._copy(game)
        player = child.players[seat]
        player.development.remove(next(c for c in player.development if c[0] == 'knight'))
        player.knights += 1
        child.largest_army_holder = award_holder([p.knights for p in child.players], 3, child.largest_army_holder)
        for i, p in enumerate(child.players):
            p.largest_army = i == child.largest_army_holder
        tiles = [i for i in range(len(child.board.tiles)) if i != child.robber_tile]
        return max(self._robber_value(child, seat, t) for t in tiles)

    def _monopoly_value(self, game, seat, resource):
        child = self._copy(game)
        for s, p in enumerate(child.players):
            if s != seat:
                child.players[seat].resources[resource] += p.resources[resource]
                p.resources[resource] = 0
        return self._score(child, seat)

    def _plenty_value(self, game, seat, resources):
        child = self._copy(game)
        for r in resources:
            child.players[seat].resources[Resource(r)] += 1
        return self._score(child, seat)

    def _development_value(self, game, seat, action):
        card = action['card']
        child = self._copy(game)
        owned = child.players[seat].development
        held = next((c for c in owned if c[0] == card), None)
        if held is not None:
            owned.remove(held)
        if card == 'knight':
            child.players[seat].development.append(held)  # _knight_value removes it
            return self._knight_value(child, seat)
        if card == 'monopoly':
            return self._monopoly_value(child, seat, Resource(action['resource']))
        if card == 'year_of_plenty':
            return self._plenty_value(child, seat, action['resources'])
        return None  # Road Building stays with the build search

    def _trade_value(self, game, seat, offer):
        child = self._copy(game)
        me, owner = child.players[seat], child.players[offer['owner']]
        for r, n in offer['give'].items():
            owner.resources[Resource(r)] -= n
            me.resources[Resource(r)] += n
        for r, n in offer['receive'].items():
            me.resources[Resource(r)] -= n
            owner.resources[Resource(r)] += n
        return self._hand_score(child, seat)

    # ------------------------------------------------------------ choice
    def choose(self, view, seat):
        actions = list(view.actions)
        kinds = {a['type'] for a in actions}
        if view.phase in ('setup_settlement', 'setup_road', 'free_roads', 'actions', 'roll'):
            return super().choose(view, seat)  # build search; non-build moves reach _fallback
        game = self._world(view)
        if 'reject_trade' in kinds:
            accept = next((a for a in actions if a['type'] == 'accept_trade'), None)
            reject = next(a for a in actions if a['type'] == 'reject_trade')
            if accept is not None and self._trade_value(game, seat, view.trade_offer) > self._hand_score(game, seat):
                return accept
            return reject
        if 'discard' in kinds:
            def kept(action):
                child = self._copy(game)
                child.players[seat].resources[Resource(action['resource'])] -= 1
                return self._hand_score(child, seat)
            return max((a for a in actions if a['type'] == 'discard'), key=kept)
        if 'move_robber' in kinds:
            tiles = [a['tile'] for a in actions if a['type'] == 'move_robber']
            return {'type': 'move_robber', 'tile': self._best_robber(game, seat, tiles)}
        if 'steal' in kinds:
            return max((a for a in actions if a['type'] == 'steal'),
                       key=lambda a: self._steal_value(game, seat, a['victim']))
        return super().choose(view, seat)

    def _fallback(self, view, seat):
        actions = list(view.actions)
        if view.phase == 'roll':
            roll = next(a for a in actions if a['type'] == 'roll')
            knight = next((a for a in actions if a['type'] == 'play_development' and a['card'] == 'knight'), None)
            if knight is None:
                return roll
            tile = view.board.tiles[view.robber_tile]
            mine = view.players[seat]
            blocked = tile.number is not None and any(v in mine.settlements or v in mine.cities for v in tile.vertices)
            game = self._world(view)
            if blocked and self._knight_value(game, seat) > self._score(game, seat):
                return knight
            return roll
        if view.phase == 'actions':
            # The turn search found nothing better than ending the turn. Score
            # each development play by the position it leads to instead of the
            # inherited random choice.
            game = self._world(view)
            best, best_value = None, self._score(game, seat)
            for candidate in actions:
                if candidate['type'] != 'play_development':
                    continue
                value = self._development_value(game, seat, candidate)
                if value is not None and value > best_value:
                    best, best_value = candidate, value
            if best is not None:
                return best
            end = next((a for a in actions if a['type'] == 'end_turn'), None)
            if end is not None:
                return end
        return super()._fallback(view, seat)


# ---------------------------------------------------------------- version 2
# Three additions to the same shared score, each switchable so a screen can
# attribute its effect. All constants were fixed before any game.
RACE_DISCOUNT = .5   # value of an award reached in k more steps: 2 points * RACE_DISCOUNT**k
OFFER_MARGIN = .25   # minimum expected score gain before an offer is sent


class SharedScorePlanner(UnifiedPlanner):
    """``planner_unified`` plus award races, opponent-aware builds and trade offers.

    - ``race``: every seat's value counts progress toward Longest Road and
      Largest Army it does not hold, so building toward an award, and an
      opponent closing in on one, both move the score.
    - ``block``: the turn search ranks builds by the shared score
      ``V(me) - LAMBDA * max V(opponent)`` instead of ``V(me)``, so a road or
      settlement that takes a leader's best site counts.
    - ``offers``: when the turn would otherwise end, propose the one-card or
      two-for-one exchange with the highest expected score gain, weighting
      each by the card-counted chance the recipient holds the requested card.
      At most one offer per turn.
    """

    race = block = offers = True

    # ------------------------------------------------------------ scoring
    def _road_lengths(self, game):
        occupied = game.occupied()
        return [longest_road(p.roads, {v for v, owner in occupied.items() if owner != s})
                for s, p in enumerate(game.players)]

    def _race_value(self, game, seat):
        player, points = game.players[seat], self.weights[0]
        bonus = 0.
        if game.longest_road_holder != seat and len(player.roads) < 15:
            lengths = self._road_lengths(game)
            holder = game.longest_road_holder
            gap = (5 if holder is None else lengths[holder] + 1) - lengths[seat]
            if 1 <= gap <= 3:
                bonus += 2 * points * RACE_DISCOUNT ** gap
        if game.largest_army_holder != seat:
            holder = game.largest_army_holder
            gap = (3 if holder is None else game.players[holder].knights + 1) - player.knights
            # Only known cards count: estimated opponent cards are placeholders (turn -1).
            held = sum(card == 'knight' and turn >= 0 for card, turn in player.development)
            if 1 <= gap <= 3:
                bonus += 2 * points * RACE_DISCOUNT ** max(gap - held, 0) * .75
        return bonus

    def _seat_value(self, game, seat):
        value = super()._seat_value(game, seat)
        return value + self._race_value(game, seat) if self.race else value

    def _leaf(self, game, seat):
        if self.block:
            mine = self._seat_value(game, seat)
            threat = max(self._seat_value(game, s) for s in range(len(game.players)) if s != seat)
            return mine - LAMBDA * threat
        if self.race:
            return self._value(game, seat) + self._race_value(game, seat)
        return self._value(game, seat)

    # ------------------------------------------------------------- offers
    def _best_offer(self, view, seat):
        game = self._world(view)
        base = self._hand_score(game, seat)
        hand = game.players[seat].resources
        best, best_gain = None, OFFER_MARGIN
        for receive in _RESOURCES:
            for give in _RESOURCES:
                if give == receive:
                    continue
                for count in (1, 2):
                    if hand[give] < count:
                        continue
                    for recipient in range(len(game.players)):
                        if recipient == seat:
                            continue
                        held = game.players[recipient].resources[receive]
                        chance = min(1., held)
                        if chance <= 0:
                            continue
                        offer = {'owner': seat, 'give': {give.value: count}, 'receive': {receive.value: 1}}
                        child = self._copy(game)
                        child.players[recipient].resources[receive] = max(held - 1, 0)
                        child.players[recipient].resources[give] += count
                        child.players[seat].resources[receive] += 1
                        child.players[seat].resources[give] -= count
                        gain = chance * (self._hand_score(child, seat) - base)
                        if gain > best_gain:
                            best, best_gain = (recipient, offer), gain
        if best is None:
            return None
        recipient, offer = best
        return {'type': 'offer_trade', 'recipient': recipient, 'give': offer['give'], 'receive': offer['receive']}

    def choose(self, view, seat):
        action = super().choose(view, seat)
        if (self.offers and view.phase == 'actions' and action.get('type') == 'end_turn'
                and self.last_offer_turn != view.turn):
            offer = self._best_offer(view, seat)
            if offer is not None:
                self.last_offer_turn = view.turn
                return offer
        return action


class RacePlanner(SharedScorePlanner):
    block = offers = False


class BlockPlanner(SharedScorePlanner):
    race = offers = False


class OfferingPlanner(SharedScorePlanner):
    race = block = False
