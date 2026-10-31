"""Offline true-world continuation diagnostic; not an attainable policy screen."""
from copy import deepcopy
import json
import argparse
import hashlib
from pathlib import Path
import random
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.tactical import public_projection


def continuation(root, action, actor, seed, turn_policy=None):
    game = deepcopy(root)
    game.rng = random.Random(seed)
    game.record_events = False
    bots = [make_policy('planner_robber', seed * 17 + s) for s in range(4)]
    planner = make_policy(turn_policy, seed * 17 + actor) if turn_policy else None
    game.apply(actor, action)
    for _ in range(20000):
        if game.winner is not None or game.turn >= root.turn + 400:
            break
        seat = game.current
        bot = planner if planner is not None and game.turn == root.turn and seat == actor else bots[seat]
        game.apply(seat, bot.choose(Observation.for_policy(game, seat, bot), seat))
    return {'winner': game.winner, 'truncated': game.winner is None, 'turns': game.turn - root.turn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole-turn', action='store_true')
    args = parser.parse_args()
    output = Path('runs/planner-disagreements-turn-v1' if args.whole_turn else 'runs/planner-disagreements-v1')
    output.mkdir(exist_ok=False)
    source = source_files()
    source['probe_script'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seed_start = 1438000 if args.whole_turn else 1437000
    for seed in range(seed_start, seed_start + 4):
        game = Game.new(['planner_robber'] * 4, seed)
        bots = [make_policy('planner_robber', seed * 17 + s) for s in range(4)]
        found = 0
        for step in range(10000):
            if game.winner is not None or game.turn >= 400:
                break
            seat = game.current
            view = Observation.for_policy(game, seat, bots[seat])
            decision_state = bots[seat].rng.getstate()
            control = bots[seat].choose(view, seat)
            if game.phase == 'actions' and found < 2:
                candidate = make_policy('planner_chain', 0)
                candidate.rng.setstate(decision_state)
                action = candidate.choose(view, seat)
                if action != control:
                    started = time.perf_counter()
                    row = {'board_seed': seed, 'step': step, 'turn': game.turn, 'seat': seat,
                           'candidate': action, 'control': control, 'replicates': [], 'immediate_scores': {}}
                    for label, choice in [('candidate', action), ('control', control)]:
                        projected = public_projection(view)
                        if choice['type'] == 'buy_development':
                            projected.players[seat].pay('development')
                            score = candidate._value(projected, seat) + candidate.purchase_value
                        elif choice['type'] in ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'end_turn'):
                            projected.apply(seat, choice)
                            score = candidate._value(projected, seat)
                        else:
                            score = None
                        row['immediate_scores'][label] = score
                    for replicate in range(4):
                        rng_seed = seed * 100 + found * 4 + replicate
                        row['replicates'].append({'seed': rng_seed,
                            'candidate': continuation(game, action, seat, rng_seed, 'planner_chain' if args.whole_turn else None),
                            'control': continuation(game, control, seat, rng_seed, 'planner_robber' if args.whole_turn else None)})
                    row['seconds'] = time.perf_counter() - started
                    row['source'] = source
                    assert source == {**source_files(), 'probe_script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                    (output / f'{seed}-{found}.json').write_text(json.dumps(row, indent=2) + '\n')
                    print(json.dumps({k: row[k] for k in ('board_seed', 'turn', 'candidate', 'control', 'seconds')}), flush=True)
                    found += 1
            game.apply(seat, control)
            if found == 2:
                break
        (output / f'{seed}-terminal.json').write_text(json.dumps({'board_seed': seed, 'states': found,
                                                                 'complete': True, 'source': source}) + '\n')


if __name__ == '__main__':
    main()
