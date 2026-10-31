"""Observation-only consistency witnesses; not posterior samples."""
from .engine import RESOURCES, Resource
from .belief import BeliefTracker, BeliefFailure, Particle



def constrain_year_of_plenty(solver, view, bank):
    """Match bank-dependent card choices revealed to the acting observer."""
    import z3
    if getattr(view, 'phase', None) not in ('roll', 'actions'):
        return
    own = view.players[view.seat]
    eligible = any(card == 'year_of_plenty' and bought < view.turn
                   for card, bought in zip(own.development, own.development_turns))
    if not eligible or view.played_development:
        return
    observed = {tuple(a['resources']) for a in view.actions
                if a['type'] == 'play_development' and a['card'] == 'year_of_plenty'}
    for i, first in enumerate(RESOURCES):
        for j in range(i, len(RESOURCES)):
            resources = (first.value, RESOURCES[j].value)
            available = z3.And(bank[i] >= 1 + (i == j), bank[j] >= 1)
            solver.add(available == (resources in observed))
        singleton = z3.And(sum(bank) == 1, bank[i] >= 1)
        solver.add(singleton == ((first.value,) in observed))


def solve_resources(view, seed=0, timeout_ms=500):
    import z3
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=seed)
    hands = [[0] * 5 for _ in range(4)]
    choices = {}
    checked = set()
    for index, event in enumerate(view.events):
        seat, action = event['seat'], event['action']
        if action['type'] == 'offer_trade':
            for resource, count in action['give'].items():
                solver.add(hands[seat][RESOURCES.index(Resource(resource))] >= count)
        if 'production_claims' in event:
            for r, resource in enumerate(RESOURCES):
                claims = [claim[resource.value] for claim in event['production_claims']]
                if not any(claims):
                    continue
                bank = 19 - sum(hand[r] for hand in hands)
                for s, claim in enumerate(claims):
                    short = z3.If(bank < claim, bank, claim) if sum(n > 0 for n in claims) == 1 else 0
                    solver.add(z3.If(sum(claims) <= bank, claim, short) ==
                               event['resource_deltas'][s].get(resource.value, 0))
        if 'resource_deltas' in event:
            for hand, delta in zip(hands, event['resource_deltas']):
                for r, resource in enumerate(RESOURCES):
                    change = delta.get(resource.value, 0)
                    if change:
                        hand[r] = hand[r] + change
        elif action['type'] in ('steal', 'discard'):
            choice = [z3.Bool(f'card_{index}_{r}') for r in range(5)]
            choices[index] = choice
            solver.add(z3.PbEq([(b, 1) for b in choice], 1))
            source = action['victim'] if action['type'] == 'steal' else seat
            for r in range(5):
                amount = z3.If(choice[r], 1, 0)
                hands[source][r] = hands[source][r] - amount
                if action['type'] == 'steal':
                    hands[seat][r] = hands[seat][r] + amount
        for s, hand in enumerate(hands):
            key = (s, tuple(('ast', n.get_id()) if isinstance(n, z3.ExprRef) else ('constant', n)
                            for n in hand), event['counts'][s])
            if key not in checked:
                solver.add(*(n >= 0 for n in hand), sum(hand) == event['counts'][s])
                checked.add(key)
        for r in range(5):
            solver.add(sum(hand[r] for hand in hands) <= 19)
        if action['type'] == 'play_development' and action['card'] == 'monopoly':
            r = RESOURCES.index(Resource(action['resource']))
            solver.add(*(hands[s][r] == 0 for s in range(4) if s != seat))
    solver.add(*(hands[view.seat][r] == view.players[view.seat].resources[resource]
                 for r, resource in enumerate(RESOURCES)))
    if getattr(view, 'phase', None) == 'actions':
        observed_trades = {(a['give'], a['receive']) for a in view.actions
                           if a['type'] == 'bank_trade'}
        own = view.players[view.seat].resources
        for give in RESOURCES:
            if own[give] < view.trade_rate(view.seat, give):
                continue
            for r, receive in enumerate(RESOURCES):
                if receive == give:
                    continue
                available = 19 - sum(hand[r] for hand in hands)
                solver.add((available > 0) == ((give.value, receive.value) in observed_trades))
    constrain_year_of_plenty(solver, view, [19 - sum(hand[r] for hand in hands) for r in range(5)])
    status = solver.check()
    if status != z3.sat:
        raise RuntimeError(f'{status}: {solver.reason_unknown()}')
    model = solver.model()
    return {index: RESOURCES[next(r for r, b in enumerate(choice)
                                  if z3.is_true(model.eval(b)))].value
            for index, choice in choices.items()}


CARDS = ('knight', 'victory_point', 'monopoly', 'year_of_plenty', 'road_building')
SUPPLY = (14, 5, 2, 2, 2)


def solve_development(view, seed=0, timeout_ms=500):
    import z3
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=seed)
    purchases = [[] for _ in range(4)]
    played = [[0] * len(CARDS) for _ in range(4)]
    draws = {}
    for index, event in enumerate(view.events):
        seat, action = event['seat'], event['action']
        if action['type'] == 'buy_development':
            card = z3.Int(f'development_{index}')
            solver.add(card >= 0, card < len(CARDS))
            if 'drawn_card' in event:
                solver.add(card == CARDS.index(event['drawn_card']))
            elif seat == view.seat:
                raise BeliefFailure('own observed purchase is missing its card identity')
            purchases[seat].append((card, event['turn']))
            draws[index] = card
        elif action['type'] == 'play_development':
            kind = CARDS.index(action['card'])
            eligible = [z3.If(card == kind, 1, 0) for card, age in purchases[seat] if age < event['turn']]
            solver.add(sum(eligible) > played[seat][kind])
            played[seat][kind] += 1
        if not event['phase'].startswith('setup'):
            owner = event['turn_owner']
            points = event['public_points'][owner] + sum(z3.If(card == 1, 1, 0)
                                                       for card, _ in purchases[owner])
            solver.add((points >= 10) == (event['winner'] == owner))
    for kind, count in enumerate(SUPPLY):
        solver.add(sum(z3.If(card == kind, 1, 0) for hand in purchases for card, _ in hand) <= count)
    for seat, player in enumerate(view.players):
        solver.add(len(purchases[seat]) - sum(played[seat]) == player.development_count)
    status = solver.check()
    if status != z3.sat:
        raise BeliefFailure(f'development constraints {status}: {solver.reason_unknown()}')
    model = solver.model()
    return {index: CARDS[model.eval(card).as_long()] for index, card in draws.items()}


def verify_witness(view, resources, development):
    """Replay concrete choices through the existing independent transition checks."""
    expected_resources = {i for i, event in enumerate(view.events)
                          if event['action']['type'] in ('steal', 'discard')
                          and 'resource_deltas' not in event}
    expected_development = {i for i, event in enumerate(view.events)
                            if event['action']['type'] == 'buy_development'}
    if set(resources) != expected_resources or set(development) != expected_development:
        raise BeliefFailure('witness choices do not match hidden actions and purchases')
    particle = Particle.empty()
    verifier = BeliefTracker(0, 1)
    for index, original in enumerate(view.events):
        event = dict(original)
        if index in resources:
            action, seat = event['action'], event['seat']
            source = action['victim'] if action['type'] == 'steal' else seat
            deltas = [{} for _ in range(4)]
            deltas[source][resources[index]] = -1
            if action['type'] == 'steal':
                deltas[seat][resources[index]] = 1
            event['resource_deltas'] = deltas
        if index in development:
            if 'drawn_card' in event and event['drawn_card'] != development[index]:
                raise BeliefFailure('witness changes an observed development card')
            event['drawn_card'] = development[index]
        if not verifier._advance(particle, event):
            raise BeliefFailure(f'witness violates observed event {index}')
    own = view.players[view.seat]
    if (particle.hands[view.seat] != own.resources or particle.development[view.seat] !=
            list(zip(own.development, own.development_turns))):
        raise BeliefFailure('witness differs from own private hand')
    for seat, player in enumerate(view.players):
        if (sum(particle.hands[seat].values()) != player.resource_count
                or len(particle.development[seat]) != player.development_count):
            raise BeliefFailure('witness differs from public card counts')
    return particle


def solve_particle(view, seed=0, timeout_ms=500):
    resources = solve_resources(view, seed, timeout_ms)
    development = solve_development(view, seed, timeout_ms)
    return verify_witness(view, resources, development)

