"""Bounded particle beliefs conditioned on a player's observed event history.

Hidden discards use a uniform-card proposal, not an opponent strategy model.
Finite particles can lose a feasible history; callers must handle BeliefFailure.
"""

from copy import deepcopy
from dataclasses import dataclass
import random

from .engine import Game, Player, RESOURCES, Resource
from .observation import Observation


class BeliefFailure(RuntimeError):
    pass


@dataclass
class Particle:
    hands: list[dict]
    development: list[list[tuple[str, int]]]
    deck: list[str]

    def __deepcopy__(self, memo):
        # Resource keys, counts and development tuples are immutable; containers
        # must be independent because transitions mutate each sampled world.
        cloned = type(self)([dict(hand) for hand in self.hands],
                            [list(cards) for cards in self.development], list(self.deck))
        memo[id(self)] = cloned
        return cloned

    @classmethod
    def empty(cls):
        return cls([{r: 0 for r in RESOURCES} for _ in range(4)], [[] for _ in range(4)],
                   ['knight'] * 14 + ['victory_point'] * 5
                   + ['monopoly', 'year_of_plenty', 'road_building'] * 2)


class BeliefTracker:
    replay_attempts = 3
    def __init__(self, seed: int, particles: int = 128):
        if particles < 1:
            raise ValueError('particles must be positive')
        self.rng = random.Random(seed)
        self.size = particles
        self.history = []
        self.population = []
        self.seat = None

    def _advance(self, particle: Particle, event: dict) -> float:
        seat, action = event['seat'], event['action']
        kind = action['type']
        hands = particle.hands
        weight = 1.0
        if kind == 'offer_trade' and any(hands[seat][Resource(r)] < n for r, n in action['give'].items()):
            return 0
        if 'production_claims' in event:
            for r in RESOURCES:
                amounts = [claim[r.value] for claim in event['production_claims']]
                bank = 19 - sum(h[r] for h in hands)
                if sum(amounts) > bank:
                    amounts = [min(n, bank) for n in amounts] if sum(n > 0 for n in amounts) == 1 else [0] * 4
                if amounts != [d.get(r.value, 0) for d in event['resource_deltas']]:
                    return 0
        if kind == 'steal' and 'resource_deltas' in event:
            victim = action['victim']
            stolen = next((Resource(r) for r, n in event['resource_deltas'][victim].items() if n == -1), None)
            total = sum(hands[victim].values())
            if stolen is None or not total:
                return 0
            weight = hands[victim][stolen] / total
            if not weight:
                return 0
        if 'resource_deltas' in event:
            for hand, delta in zip(hands, event['resource_deltas']):
                for resource, amount in delta.items():
                    hand[Resource(resource)] += amount
        elif kind in {'steal', 'discard'}:
            source = action['victim'] if kind == 'steal' else seat
            cards = [r for r, n in hands[source].items() for _ in range(n)]
            if not cards:
                return 0
            resource = self.rng.choice(cards)
            hands[source][resource] -= 1
            if kind == 'steal':
                hands[seat][resource] += 1
        if any(n < 0 for hand in hands for n in hand.values()):
            return 0
        if [sum(h.values()) for h in hands] != event['counts']:
            return 0
        if any(sum(h[r] for h in hands) > 19 for r in RESOURCES):
            return 0
        if kind == 'play_development' and action['card'] == 'monopoly':
            if any(h[Resource(action['resource'])] for s, h in enumerate(hands) if s != seat):
                return 0
        if kind == 'buy_development':
            if not particle.deck:
                return 0
            if 'drawn_card' in event:
                card = event['drawn_card']
                weight = particle.deck.count(card) / len(particle.deck)
                if not weight:
                    return 0
            else:
                card = self.rng.choice(particle.deck)
            particle.deck.remove(card)
            particle.development[seat].append((card, event['turn']))
        elif kind == 'play_development':
            eligible = next((c for c in particle.development[seat]
                             if c[0] == action['card'] and c[1] < event['turn']), None)
            if eligible is None:
                return 0
            particle.development[seat].remove(eligible)
        if not event['phase'].startswith('setup'):
            owner = event['turn_owner']
            points = event['public_points'][owner] + sum(c == 'victory_point' for c, _ in particle.development[owner])
            if (points >= 10) != (event['winner'] == owner):
                return 0
        return weight

    def update(self, view: Observation) -> None:
        events = list(view.events)
        prefix = (self.seat == view.seat and len(events) >= len(self.history)
                  and events[:len(self.history)] == self.history and self.population)
        for attempt in range(self.replay_attempts):
            population = deepcopy(self.population) if prefix and attempt == 0 else [Particle.empty() for _ in range(self.size)]
            start = len(self.history) if prefix and attempt == 0 else 0
            for event in events[start:]:
                weights = [self._advance(p, event) for p in population]
                if not any(weights):
                    population = []
                    break
                # Resample only when observations distinguish particles; deterministic
                # public events must not needlessly impoverish the population.
                if len(set(weights)) > 1:
                    population = [deepcopy(p) for p in self.rng.choices(population, weights, k=self.size)]
            own = view.players[view.seat]
            survivors = [p for p in population
                         if p.hands[view.seat] == own.resources
                         and p.development[view.seat] == list(zip(own.development, own.development_turns))
                         and all(sum(p.hands[s].values()) == v.resource_count
                                 and len(p.development[s]) == v.development_count
                                 for s, v in enumerate(view.players))]
            if survivors:
                self.population = survivors
                self.history = deepcopy(events)
                self.seat = view.seat
                return
        self.population = []
        raise BeliefFailure('no particles consistent with the observed history')

    def sample(self, view: Observation, seed: int) -> Game:
        if self.seat != view.seat or self.history != list(view.events) or not self.population:
            self.update(view)
        rng = random.Random(seed)
        for _ in range(32):
            particle = rng.choice(self.population)
            players = [Player(str(s), dict(particle.hands[s]), set(v.roads),
                              set(v.settlements), set(v.cities),
                              s == view.longest_road_holder, list(particle.development[s]),
                              v.knights, s == view.largest_army_holder)
                       for s, v in enumerate(view.players)]
            deck = list(particle.deck)
            rng.shuffle(deck)
            game = Game(view.board, players, rng, current=view.seat, phase=view.phase,
                        setup_index=view.setup_index, pending_setup_vertex=view.pending_setup_vertex,
                        rolled=view.rolled, robber_tile=view.robber_tile, turn=view.turn,
                        discard_remaining=dict(view.discard_remaining), robber_actor=view.robber_actor,
                        longest_road_holder=view.longest_road_holder, deck=deck,
                        played_development=view.played_development, return_phase=view.return_phase,
                        free_roads=view.free_roads, largest_army_holder=view.largest_army_holder,
                        trade_offer=deepcopy(view.trade_offer), record_events=False)
            game._update_longest_road()
            if game.legal_actions(view.seat) == list(view.actions):
                return game
        raise BeliefFailure('particles do not reproduce observed legal actions')
