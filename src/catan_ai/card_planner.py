"""Approximate public-history resource estimates for Monopoly selection."""
from .engine import Resource
from .planner import ProductionRobberPlanner, WEIGHTS


_RESOURCES = tuple(Resource)
_FOLDS = {}  # performance only: resumable folds keyed by the first event object


def _fold(hands, event):
    action, seat = event['action'], event['seat']
    if 'resource_deltas' in event:
        for hand, delta in zip(hands, event['resource_deltas']):
            for resource, count in delta.items():
                hand[Resource(resource)] += count
    elif action['type'] in ('steal', 'discard'):
        source = action['victim'] if action['type'] == 'steal' else seat
        total = sum(hands[source].values())
        if total:
            for resource in _RESOURCES:
                amount = hands[source][resource] / total
                hands[source][resource] -= amount
                if action['type'] == 'steal':
                    hands[seat][resource] += amount
    for hand, count in zip(hands, event['counts']):
        for resource in _RESOURCES:
            hand[resource] = max(0.0, hand[resource])
        total = sum(hand.values())
        for resource in _RESOURCES:
            hand[resource] = hand[resource] * count / total if total else count / len(_RESOURCES)


def resource_estimates(events):
    """Card-counted hands from public history: a left fold over ``events``.

    Performance only: when ``events`` extends a sequence seen before — every folded
    event is the identical object, in order (``Game.observed_view_events`` returns
    read-only shared events) — the fold resumes from the stored state instead of
    restarting; the arithmetic and its order are unchanged. Stored entries hold the
    event objects themselves, so identities cannot be reused.
    """
    events = list(events)
    start, hands = 0, [{r: 0.0 for r in _RESOURCES} for _ in range(4)]
    if events:
        entry = _FOLDS.get(id(events[0]))
        if entry is not None:
            seen, state = entry
            # resume only if every event already folded is the identical object, in order
            if len(seen) <= len(events) and all(a is b for a, b in zip(seen, events)):
                start, hands = len(seen), [dict(h) for h in state]
    for event in events[start:]:
        _fold(hands, event)
    if events:
        if len(_FOLDS) > 64:
            _FOLDS.clear()
        _FOLDS[id(events[0])] = (tuple(events), [dict(h) for h in hands])
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
