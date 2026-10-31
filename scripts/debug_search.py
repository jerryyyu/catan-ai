"""Replay recorded choices and audit search without rerunning policy selection.

Privileged-state continuation probes are offline diagnostics, never policy input.
"""
import argparse
import hashlib
import importlib.util
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.search import MonteCarloBot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('runs/mc-dev-v1.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/mc-debug-v1.json'))
    parser.add_argument('--probes', type=int, default=128)
    parser.add_argument('--shortlist-source', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.input.read_text())
    source = source_files()
    if source != receipt['source']:
        raise ValueError('replay requires the original policy/engine source')
    shortlist_class = MonteCarloBot
    if args.shortlist_source:
        import sys
        spec = importlib.util.spec_from_file_location('catan_ai.repaired_shortlist', args.shortlist_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        shortlist_class = module.MonteCarloBot
    counters = Counter()
    panels = {}
    roots = []
    omissions = []
    for row in receipt['games']:
        panel = next(p for p in receipt['config']['panels'] if p['name'] == row['panel'])
        names = list(panel['opponents'])
        seat, seed = row['seat'], row['seed']
        names.insert(seat, 'heuristic')
        bots = [make_policy(n, seed * 17 + s + 1) for s, n in enumerate(names)]
        search = MonteCarloBot(seed * 17 + seat + 1)
        search.baseline = bots[seat]
        shortlist = shortlist_class(seed * 17 + seat + 1)
        shortlist.baseline = bots[seat]
        game = Game.new([f'{n}-{s}' for s, n in enumerate(names)], seed)
        records = iter(row['candidate']['search'][str(seat)])
        while game.winner is None:
            actor = game.current
            view = Observation.from_game(game, actor)
            action = bots[actor].choose(view, actor)
            if actor == seat:
                record = next(records)
                assert record['baseline'] == action and record['turn'] == game.turn
                action = record['chosen']
                if game.phase == 'actions':
                    candidates = shortlist._candidates(view, record['baseline'])
                    assert candidates[0] == record['baseline']
                    assert len(candidates) <= shortlist.config.candidates
                    assert all(a in view.actions for a in candidates[1:])
                    trades = [a for a in view.actions if a['type'] == 'bank_trade']
                    useful = search.baseline._useful_trade(view, seat, trades)
                    if useful:
                        counters['useful_trade_available'] += 1
                        if useful not in candidates:
                            counters['useful_trade_omitted'] += 1
                            if len(omissions) < 5:
                                omissions.append({'panel': row['panel'], 'seed': seed, 'seat': seat,
                                                  'turn': game.turn, 'missing': useful,
                                                  'shortlist': candidates})
                    if record['reason'] == 'baseline_selected':
                        counters['baseline_selected'] += 1
                        values = record['values']
                        counters['baseline_tied'] += sum(v['wins'] == values[0]['wins'] for v in values) > 1
                    if record['lcb'] is not None:
                        mean = sum(record['confirmation_differences']) / len(record['confirmation_differences'])
                        counters['confirmation'] += 1
                        counters['positive_rejected'] += mean > 0 and record['lcb'] <= 0
                    if record['reason'] == 'confirmed':
                        world = deepcopy(game)
                        world.events = []
                        world.record_events = False
                        deltas = []
                        truncated = 0
                        for probe in range(args.probes):
                            future_seed = 990000 + probe
                            world.rng = random.Random(future_seed)
                            world.deck = sorted(game.deck)
                            world.rng.shuffle(world.deck)
                            a = search._rollout(world, action, seat, future_seed)
                            b = search._rollout(world, record['baseline'], seat, future_seed)
                            if a is None or b is None:
                                truncated += 1
                            else:
                                deltas.append(a - b)
                        roots.append({'panel': row['panel'], 'seed': seed, 'seat': seat,
                                      'turn': game.turn, 'baseline': record['baseline'], 'chosen': action,
                                      'original_belief_gain': sum(record['confirmation_differences']) / 32,
                                      'privileged_hand_probe_gain': sum(deltas) / len(deltas) if deltas and not truncated else None,
                                      'completed_pairs': len(deltas), 'truncated_pairs': truncated,
                                      'paired_differences': deltas})
                counters['decisions'] += 1
            game.apply(actor, action)
        assert next(records, None) is None
        assert game.winner == row['candidate']['winner'] and game.turn == row['candidate']['turns']
        assert tuple(p.points for p in game.players) == tuple(row['candidate']['points'])
        counters['replayed_games'] += 1
        p = panels.setdefault(row['panel'], Counter())
        p['search_wins'] += game.winner == seat
        p['control_wins'] += row['baseline']['winner'] == seat
        print(json.dumps({'replayed': counters['replayed_games'], 'seed': seed, 'seat': seat}), flush=True)
    result = {'source': source, 'panels': panels, 'counters': counters, 'omission_examples': omissions,
              'shortlist_source_sha256': hashlib.sha256(args.shortlist_source.read_bytes()).hexdigest() if args.shortlist_source else None,
              'override_probes': roots, 'confirmation_threshold': math.sqrt(2 * math.log(20) / 32),
              'probe_note': 'Offline true-hand and remaining-deck-multiset diagnostic, fresh future RNG; all continuations heuristic. No strength claim.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'counters': counters, 'panels': panels,
                      'probes': [{k: v for k, v in r.items() if k != 'paired_differences'} for r in roots]}, indent=2))


if __name__ == '__main__':
    main()
