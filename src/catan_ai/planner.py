"""Bounded public-information turn search with an economic leaf evaluation."""
from copy import deepcopy
import heapq
import math

from .bots import HeuristicBot
from .engine import Resource
from .tactical import public_projection


WEIGHTS = {Resource.BRICK: 1.0, Resource.LUMBER: 1.0, Resource.WOOL: .8,
           Resource.GRAIN: 1.3, Resource.ORE: 1.5}


class EconomicPlanner(HeuristicBot):
    purchase_value = 5.5
    buy_after_root = False
    def _income(self, game, seat):
        player = game.players[seat]
        income = {r: 0.0 for r in Resource}
        for tile in game.board.tiles:
            if tile.number is not None:
                count = sum((v in player.settlements) + 2 * (v in player.cities) for v in tile.vertices)
                income[tile.resource] += count * (6 - abs(7 - tile.number))
        return income

    def _geometry(self, game, seat):
        player = game.players[seat]
        occupied = game.occupied()
        enemy_edges = {edge for s, p in enumerate(game.players) if s != seat for edge in p.roads}
        starts = set(player.settlements | player.cities)
        starts.update(v for edge in player.roads for v in edge if occupied.get(v, seat) == seat)
        distance = {v: 0 for v in starts}
        queue = [(0, v) for v in starts]
        heapq.heapify(queue)
        while queue:
            cost, vertex = heapq.heappop(queue)
            if cost != distance[vertex] or cost >= 3 or occupied.get(vertex, seat) != seat:
                continue
            for neighbor in game.board.neighbors[vertex]:
                edge = tuple(sorted((vertex, neighbor)))
                if edge in enemy_edges:
                    continue
                candidate = cost + (edge not in player.roads)
                if candidate < distance.get(neighbor, math.inf):
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        sites = []
        for vertex, distance_to in distance.items():
            if vertex in occupied or any(n in occupied for n in game.board.neighbors[vertex]):
                continue
            production = sum((6 - abs(7 - t.number)) * WEIGHTS[t.resource]
                             for t in game.board.tiles if vertex in t.vertices and t.number is not None)
            sites.append((5 + .3 * production) / (1 + distance_to))
        sites.sort(reverse=True)
        return sum(value * weight for value, weight in zip(sites, (1, .4, .15)))

    def _value(self, game, seat):
        player = game.players[seat]
        if player.points >= 10:
            return 100000.0
        income = self._income(game, seat)
        production = sum(WEIGHTS[r] * (n + 2 * math.sqrt(n)) for r, n in income.items())
        hand = sum(WEIGHTS[r] * (1.6 * min(n, 3) + .45 * max(n - 3, 0))
                   for r, n in player.resources.items())
        discard = .5 * max(sum(player.resources.values()) - 7, 0)
        ports = sum(max(income[r] - 5, 0) * (4 / game.trade_rate(seat, r) - 1) for r in Resource)
        development = sum(6 if card == 'victory_point' else 3 for card, _ in player.development)
        return (12 * player.points + .55 * production + hand - discard
                + self._geometry(game, seat) + .5 * ports + development)

    def choose(self, view, seat):
        legal = view.legal_actions(seat)
        root = public_projection(view)
        if self.buy_after_root and sum(e['action']['type'] == 'buy_development'
                                       for e in view.events) < 25:
            # Purchases are terminal pay-only leaves; no placeholder card is drawn.
            root.deck = ['unknown']
        if view.phase == 'setup_settlement':
            def placement(action):
                child = deepcopy(root, {id(root.board): root.board})
                child.players[seat].settlements.add(action['vertex'])
                return self._value(child, seat)
            return max(legal, key=placement)
        if view.phase not in {'actions', 'free_roads'}:
            return super().choose(view, seat)
        base = self._value(root, seat)
        best_action = next((a for a in legal if a['type'] == 'end_turn'), None)
        best_score = base if best_action else -math.inf
        frontier = [(root, None, legal)]
        nodes = 0
        for depth in range(3):
            successors = []
            for game, first, actions in frontier:
                for action in actions:
                    kind = action['type']
                    build = kind in {'build_city', 'build_settlement', 'build_road'}
                    acquire = kind == 'bank_trade' or kind == 'play_development' and action['card'] in {'year_of_plenty', 'road_building'}
                    buy = kind == 'buy_development'
                    if not (build or buy and (depth == 0 or self.buy_after_root)
                            or depth == 0 and acquire):
                        continue
                    if nodes >= 128:
                        break
                    nodes += 1
                    child = deepcopy(game, {id(game.board): game.board})
                    if buy:
                        child.players[seat].pay('development')
                    else:
                        child.apply(seat, action)
                    score = self._value(child, seat) + (self.purchase_value if buy else 0) - .01 * (depth + 1)
                    first_action = first or action
                    if score > best_score:
                        best_action, best_score = first_action, score
                    if not buy and child.winner is None:
                        successors.append((score, child, first_action))
            successors.sort(key=lambda row: row[0], reverse=True)
            frontier = [(game, first, game.legal_actions(seat)) for _, game, first in successors[:8]]
        if best_action is not None and best_action['type'] != 'end_turn':
            return best_action
        # Special development effects and bilateral negotiation retain the
        # baseline's legal handling until their economic model is measured.
        if any(a['type'] == 'play_development' for a in legal):
            return super().choose(view, seat)
        return best_action or super().choose(view, seat)


class DevelopmentPlanner(EconomicPlanner):
    # The DEV arm tests whether charging the full hand cost while giving little
    # credit for army progress suppresses useful development-card purchases.
    purchase_value = 8.0


class CoveragePlanner(DevelopmentPlanner):
    def choose(self, view, seat):
        if view.phase != 'setup_settlement':
            return super().choose(view, seat)
        root = public_projection(view)
        def opening(action):
            child = deepcopy(root, {id(root.board): root.board})
            child.players[seat].settlements.add(action['vertex'])
            income = self._income(child, seat)
            return (sum(income.values()) + 4 * sum(value > 0 for value in income.values()),
                    self._geometry(child, seat))
        return max(view.actions, key=opening)


class ProductionRobberPlanner(DevelopmentPlanner):
    @staticmethod
    def _robber_score(view, seat, tile_index):
        tile = view.board.tiles[tile_index]
        pips = 0 if tile.number is None else 6 - abs(7 - tile.number)
        score = 0.0
        for owner, player in enumerate(view.players):
            buildings = sum((v in player.settlements) + 2 * (v in player.cities) for v in tile.vertices)
            weight = -2.0 if owner == seat else .5 + .1 * player.public_points
            score += pips * buildings * weight
        return score


class DevelopmentSequencePlanner(ProductionRobberPlanner):
    buy_after_root = True
    needs_history = True


class OfferPlanner(ProductionRobberPlanner):
    """ProductionRobberPlanner plus one economically-priced player offer per turn.

    Board row 6, first test. When the turn search finds nothing better than
    ending the turn, try every one-for-one exchange the hand allows, value the
    hypothetical accepted state with the same economic leaf, and send the best
    offer if it improves on the current position by more than ``offer_margin``.
    The recipient is the opponent with the fewest public points, so an accepted
    trade helps the trailing player rather than the leader. Acceptance is the
    partner's choice; a rejected offer costs nothing but the one offer per turn.
    Only the public projection and the planner's own hand are used.
    """

    offer_margin = .25

    def choose(self, view, seat):
        action = super().choose(view, seat)
        if (action.get('type') != 'end_turn' or view.phase != 'actions'
                or self.last_offer_turn == view.turn):
            return action
        offer = self._best_offer(view, seat)
        if offer is None:
            return action
        self.last_offer_turn = view.turn
        return offer

    def _best_offer(self, view, seat):
        root = public_projection(view)
        hand = root.players[seat].resources
        base = self._value(root, seat)
        best, best_gain = None, self.offer_margin
        for give in Resource:
            if hand[give] < 1:
                continue
            for receive in Resource:
                if receive == give:
                    continue
                child = deepcopy(root, {id(root.board): root.board})
                child.players[seat].resources[give] -= 1
                child.players[seat].resources[receive] += 1
                gain = self._value(child, seat) - base
                if gain > best_gain:
                    best, best_gain = (give, receive), gain
        if best is None:
            return None
        others = [p for p in view.players if p.seat != seat] if hasattr(view.players[0], 'seat') else None
        if others is None:
            points = [(view.players[s].public_points, s) for s in range(4) if s != seat]
        else:
            points = [(p.public_points, p.seat) for p in others]
        lowest = min(points)[0]
        recipient = self.rng.choice([s for score, s in points if score == lowest])
        return {'type': 'offer_trade', 'recipient': recipient,
                'give': {best[0].value: 1}, 'receive': {best[1].value: 1}}
