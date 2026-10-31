"""Approximate public-history resource estimates for Monopoly selection."""
from .engine import Resource
from .planner import ProductionRobberPlanner, WEIGHTS


def resource_estimates(events):
    hands = [{r: 0.0 for r in Resource} for _ in range(4)]
    for event in events:
        action, seat = event['action'], event['seat']
        if 'resource_deltas' in event:
            for hand, delta in zip(hands, event['resource_deltas']):
                for resource, count in delta.items():
                    hand[Resource(resource)] += count
        elif action['type'] in ('steal', 'discard'):
            source = action['victim'] if action['type'] == 'steal' else seat
            total = sum(hands[source].values())
            if total:
                for resource in Resource:
                    amount = hands[source][resource] / total
                    hands[source][resource] -= amount
                    if action['type'] == 'steal':
                        hands[seat][resource] += amount
        for hand, count in zip(hands, event['counts']):
            for resource in Resource:
                hand[resource] = max(0.0, hand[resource])
            total = sum(hand.values())
            for resource in Resource:
                hand[resource] = hand[resource] * count / total if total else count / len(Resource)
    return hands


class CardPlanner(ProductionRobberPlanner):
    needs_history = True

    def choose(self, view, seat):
        action = super().choose(view, seat)
        if action['type'] != 'play_development' or action['card'] != 'monopoly':
            return action
        estimates = resource_estimates(view.events)
        choices = [a for a in view.actions if a['type'] == 'play_development' and a['card'] == 'monopoly']
        def value(candidate):
            resource = Resource(candidate['resource'])
            return WEIGHTS[resource] * sum(hand[resource] for s, hand in enumerate(estimates) if s != seat)
        return max(choices, key=value)
