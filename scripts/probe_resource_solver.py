"""Resource-only consistency witnesses for preserved observation failures."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import z3
from catan_ai.engine import RESOURCES, Resource
from catan_ai.experiments import source_files


CAPTURES = ['external-belief-failure-v1', 'external-timeout-regression-v1',
            'external-belief-regression-v4', 'external-belief-regression-v5',
            'external-belief-regression-v6', 'external-belief-regression-v8']


def solve_resources(view):
    solver = z3.Solver()
    solver.set(timeout=500, random_seed=0)
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
            choice = z3.Int(f'card_{index}')
            choices[index] = choice
            solver.add(choice >= 0, choice < 5)
            source = action['victim'] if action['type'] == 'steal' else seat
            for r in range(5):
                amount = z3.If(choice == r, 1, 0)
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
    status = solver.check()
    if status != z3.sat:
        raise RuntimeError(f'{status}: {solver.reason_unknown()}')
    model = solver.model()
    return {index: RESOURCES[model.eval(choice).as_long()].value for index, choice in choices.items()}


def verify_resources(view, witness):
    hands = [{r.value: 0 for r in RESOURCES} for _ in range(4)]
    for index, event in enumerate(view.events):
        seat, action = event['seat'], event['action']
        if action['type'] == 'offer_trade':
            assert all(hands[seat][r] >= n for r, n in action['give'].items())
        if 'production_claims' in event:
            for resource in RESOURCES:
                r = resource.value
                claims = [c[r] for c in event['production_claims']]
                bank = 19 - sum(hand[r] for hand in hands)
                if sum(claims) > bank:
                    claims = ([min(n, bank) for n in claims] if sum(n > 0 for n in claims) == 1
                              else [0] * 4)
                assert claims == [d.get(r, 0) for d in event['resource_deltas']]
        if 'resource_deltas' in event:
            for hand, delta in zip(hands, event['resource_deltas']):
                for r, n in delta.items():
                    hand[r] += n
        elif action['type'] in ('steal', 'discard'):
            source = action['victim'] if action['type'] == 'steal' else seat
            resource = witness[index]
            assert hands[source][resource] > 0
            hands[source][resource] -= 1
            if action['type'] == 'steal':
                hands[seat][resource] += 1
        assert all(n >= 0 for hand in hands for n in hand.values())
        assert [sum(hand.values()) for hand in hands] == event['counts']
        assert all(sum(hand[r.value] for hand in hands) <= 19 for r in RESOURCES)
        if action['type'] == 'play_development' and action['card'] == 'monopoly':
            assert all(hands[s][action['resource']] == 0 for s in range(4) if s != seat)
    assert hands[view.seat] == {r.value: n for r, n in view.players[view.seat].resources.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in CAPTURES:
        data = (Path('runs') / name / 'observation.pkl').read_bytes()
        view = pickle.loads(data)
        started = time.perf_counter()
        witness, error = None, None
        try:
            witness = solve_resources(view)
            verify_resources(view, witness)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        elapsed = time.perf_counter() - started
        row = {'capture': name, 'input_sha256': hashlib.sha256(data).hexdigest(),
               'seconds': elapsed, 'error': error, 'pass': error is None and elapsed <= 1}
        (args.output / f'{name}.json').write_text(json.dumps({**row, 'witness': witness}, indent=2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / 'summary.json').write_text(json.dumps({
        'source': source_files(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'solver_version': z3.get_version_string(), 'rows': rows,
        'pass': all(row['pass'] for row in rows)}, indent=2))


if __name__ == '__main__':
    main()
