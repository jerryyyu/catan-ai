"""Particle proposals constrained by the observer's already-known later events.

This is retrospective conditioning within observed history, never future game
information. Proposal normalization retains the uniform-card hidden-action prior.
"""
from .belief import BeliefTracker
from .engine import RESOURCES, Resource
from .development_constraints import draw_constraints, feasible_draw
from .resource_constraints import subset_constraints, resource_upper_before, player_group_constraints


class GuidedBeliefTracker(BeliefTracker):
    def __init__(self, seed, particles=128):
        super().__init__(seed, particles)
        self.draw_cache = {}

    def update(self, view):
        self.draw_cache.clear()
        self.draw_requirements = draw_constraints(view.events)
        upper_before = resource_upper_before(view.events)
        required = [{r: 0 for r in RESOURCES} for _ in range(4)]
        required[view.seat] = dict(view.players[view.seat].resources)
        self.minimum_after = {}
        for event in reversed(view.events):
            self.minimum_after[id(event)] = [dict(hand) for hand in required]
            action, seat = event['action'], event['seat']
            if 'resource_deltas' in event:
                for hand, delta in zip(required, event['resource_deltas']):
                    for r in RESOURCES:
                        hand[r] = max(0, hand[r] - delta.get(r.value, 0))
            elif action['type'] == 'steal':
                forced = next((r for r in RESOURCES
                               if required[seat][r] > upper_before[id(event)][seat][r]), None)
                if forced is not None:
                    required[seat][forced] -= 1
                    required[action['victim']][forced] += 1
                else:
                    for r in RESOURCES:
                        required[seat][r] = max(0, required[seat][r] - 1)
            if action['type'] == 'offer_trade':
                for resource, count in action['give'].items():
                    r = Resource(resource)
                    required[seat][r] = max(required[seat][r], count)
        self.subset_after = subset_constraints(view, self.minimum_after)
        self.group_after = player_group_constraints(view.events, self.minimum_after)
        self.single_after = {key: [(s, r, n) for s, hand in enumerate(hands)
                                   for r, n in hand.items() if n > 0]
                             for key, hands in self.minimum_after.items()}
        return super().update(view)

    def _advance(self, particle, event):
        minimum = self.minimum_after[id(event)]
        singles = (self.single_after[id(event)] if hasattr(self, 'single_after') else
                   [(s, r, n) for s, hand in enumerate(minimum) for r, n in hand.items() if n > 0])
        subsets = getattr(self, 'subset_after', {}).get(id(event), ())
        groups = getattr(self, 'group_after', {}).get(id(event), ())
        action, seat = event['action'], event['seat']
        kind = action['type']
        proposal_weight = None
        if kind == 'buy_development' and 'drawn_card' not in event:
            key = (id(event), tuple(tuple(cards) for cards in particle.development), tuple(particle.deck))
            if key not in self.draw_cache:
                allowed = [card for card in dict.fromkeys(particle.deck)
                           if feasible_draw(particle, seat, card, event['turn'],
                                            self.draw_requirements[id(event)])]
                weights = [particle.deck.count(card) for card in allowed]
                self.draw_cache[key] = (allowed, weights)
            allowed, weights = self.draw_cache[key]
            if not allowed:
                return 0
            proposal_weight = sum(weights) / len(particle.deck)
            card = self.rng.choices(allowed, weights, k=1)[0]
            event = {**event, 'drawn_card': card}
        if kind in {'steal', 'discard'} and 'resource_deltas' not in event:
            source = action['victim'] if kind == 'steal' else seat
            hand = particle.hands[source]
            total = sum(hand.values())
            allowed, weights = [], []
            for resource, count in hand.items():
                if not count:
                    continue
                feasible = all(
                    particle.hands[s][r] - int(s == source and r == resource)
                    + int(kind == 'steal' and s == seat and r == resource) >= bound
                    for s, r, bound in singles)
                if feasible:
                    feasible = all(sum(particle.hands[s][r] for r in group)
                                   - int(s == source and resource in group)
                                   + int(kind == 'steal' and s == seat and resource in group) >= bound
                                   for s, group, bound in subsets)
                if feasible:
                    feasible = all(sum(particle.hands[s][r] for s in owners)
                                   - int(source in owners and resource == r)
                                   + int(kind == 'steal' and seat in owners and resource == r) >= bound
                                   for owners, r, bound in groups)
                if feasible:
                    allowed.append(resource)
                    weights.append(count)
            if not allowed:
                return 0
            resource = self.rng.choices(allowed, weights, k=1)[0]
            deltas = [{} for _ in range(4)]
            deltas[source][resource.value] = -1
            if kind == 'steal':
                deltas[seat][resource.value] = 1
            result = super()._advance(particle, {**event, 'resource_deltas': deltas})
            return sum(weights) / total if result else 0
        weight = super()._advance(particle, event)
        if any(particle.hands[s][r] < bound for s, r, bound in singles):
            return 0
        if any(sum(particle.hands[s][r] for r in group) < bound for s, group, bound in subsets):
            return 0
        if any(sum(particle.hands[s][r] for s in owners) < bound for owners, r, bound in groups):
            return 0
        return proposal_weight if weight and proposal_weight is not None else weight
