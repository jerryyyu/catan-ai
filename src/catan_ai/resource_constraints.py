"""Necessary resource-subset bounds from already observed event suffixes."""
from .engine import RESOURCES


SUBSETS = tuple(tuple(r for i, r in enumerate(RESOURCES) if mask & (1 << i))
                for mask in range(1, 1 << len(RESOURCES)))


def resource_upper_before(events):
    """Conservative per-resource maxima using only each observed prefix."""
    upper = [{r: 0 for r in RESOURCES} for _ in range(4)]
    result = {}
    for event in events:
        result[id(event)] = [dict(hand) for hand in upper]
        if 'resource_deltas' in event:
            for hand, delta in zip(upper, event['resource_deltas']):
                for r in RESOURCES:
                    hand[r] = max(0, hand[r] + delta.get(r.value, 0))
        elif event['action']['type'] == 'steal':
            seat, victim = event['seat'], event['action']['victim']
            for r in RESOURCES:
                upper[seat][r] += int(upper[victim][r] > 0)
        for hand, count in zip(upper, event['counts']):
            for r in RESOURCES:
                hand[r] = min(hand[r], count, 19)
    return result


def subset_constraints(view, individual_after):
    required = [[0] * len(SUBSETS) for _ in range(4)]
    required[view.seat] = [sum(view.players[view.seat].resources[r] for r in group)
                           for group in SUBSETS]
    result = {}
    for event in reversed(view.events):
        singles = individual_after[id(event)]
        # Only keep bounds stronger than the existing per-resource constraints.
        extras = []
        for seat in range(4):
            candidates = []
            for group, bound in zip(SUBSETS, required[seat]):
                slack = bound - sum(singles[seat][r] for r in group)
                if len(group) <= 1 or slack <= 0:
                    continue
                if len(group) == len(RESOURCES) and bound <= event['counts'][seat]:
                    continue
                candidates.append((group, bound, slack))
            kept = []
            for group, bound, slack in sorted(candidates, key=lambda row: len(row[0])):
                if any(other_slack >= slack and all(r in group for r in other)
                       for other, _, other_slack in kept):
                    continue
                kept.append((group, bound, slack))
            extras.extend((seat, group, bound) for group, bound, _ in kept)
        result[id(event)] = extras
        if 'resource_deltas' in event:
            for seat, delta in enumerate(event['resource_deltas']):
                for i, group in enumerate(SUBSETS):
                    required[seat][i] = max(0, required[seat][i] -
                                            sum(delta.get(r.value, 0) for r in group))
        elif event['action']['type'] == 'steal':
            seat = event['seat']
            required[seat] = [max(0, n - 1) for n in required[seat]]
        if event['action']['type'] == 'offer_trade':
            seat = event['seat']
            give = event['action']['give']
            required[seat] = [max(bound, sum(give.get(r.value, 0) for r in group))
                              for group, bound in zip(SUBSETS, required[seat])]
    return result


def player_group_constraints(events, individual_after):
    """Transfers within a player group conserve each resource's total."""
    groups = sorted((tuple(s for s in range(4) if mask & (1 << s))
                     for mask in range(1, 16)), key=len)
    required = [{r: 0 for r in RESOURCES} for _ in groups]
    result = {}
    for event in reversed(events):
        singles = individual_after[id(event)]
        extras = []
        kept = {r: [] for r in RESOURCES}
        for group, bounds in zip(groups, required):
            for r in RESOURCES:
                base = sum(singles[s][r] for s in group)
                bounds[r] = max(bounds[r], base)
                slack = bounds[r] - base
                if slack > 0 and not any(other_slack >= slack and all(s in group for s in other)
                                         for other, other_slack in kept[r]):
                    extras.append((group, r, bounds[r]))
                    kept[r].append((group, slack))
        result[id(event)] = extras
        if 'resource_deltas' in event:
            for group, bounds in zip(groups, required):
                for r in RESOURCES:
                    bounds[r] = max(0, bounds[r] - sum(event['resource_deltas'][s].get(r.value, 0)
                                                      for s in group))
        elif event['action']['type'] == 'steal':
            seat, victim = event['seat'], event['action']['victim']
            for group, bounds in zip(groups, required):
                if seat in group and victim not in group:
                    for r in RESOURCES:
                        bounds[r] = max(0, bounds[r] - 1)
    return result
