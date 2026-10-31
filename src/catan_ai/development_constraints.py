"""Necessary hidden-draw constraints from an already observed event suffix."""
from collections import Counter


def draw_constraints(events):
    """Bound card requirements at each later play, respecting purchase age."""
    constraints = {}
    for index, event in enumerate(events):
        if event['action']['type'] != 'buy_development':
            continue
        buys = [[] for _ in range(4)]
        plays = [Counter() for _ in range(4)]
        prefixes = []
        for later in events[index + 1:]:
            seat, action = later['seat'], later['action']
            if action['type'] == 'buy_development':
                buys[seat].append((later['turn'], later.get('drawn_card')))
            elif action['type'] == 'play_development':
                plays[seat][action['card']] += 1
                eligible = [card for turn, card in buys[seat] if turn < later['turn']]
                prefixes.append((seat, later['turn'], dict(plays[seat]),
                                 Counter(c for c in eligible if c is not None), eligible.count(None)))
        known = [Counter(card for _, card in purchases if card is not None) for purchases in buys]
        constraints[id(event)] = (prefixes, plays, known)
    return constraints


def feasible_draw(particle, seat, card, turn, constraints):
    """Reject only necessary-bound violations; this is not a posterior sampler."""
    prefixes, plays, known = constraints
    held = [list(cards) for cards in particle.development]
    held[seat].append((card, turn))
    deck = Counter(particle.deck)
    deck[card] -= 1
    for owner, deadline, needed, fixed, unknown in prefixes:
        available = Counter(c for c, age in held[owner] if age < deadline)
        deficit = sum(max(0, count - available[c] - fixed[c]) for c, count in needed.items())
        if deficit > unknown:
            return False
    required = Counter()
    for owner in range(4):
        available = Counter(c for c, _ in held[owner])
        required.update(known[owner])
        required.update({c: max(0, count - available[c] - known[owner][c])
                         for c, count in plays[owner].items()})
    return all(deck[c] >= count for c, count in required.items())
