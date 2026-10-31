"""Offline simulator labels; hidden state never enters the action feature producer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.paired_value import shortlist


def continuation(root, action, actor, names, seed):
    game = deepcopy(root, {id(root.board): root.board})
    game.rng = random.Random(seed)
    random.Random(seed ^ 0x917ABC).shuffle(game.deck)
    game.record_events, game.events = False, []
    policies = list(names)
    policies[actor] = 'planner_robber'
    bots = [make_policy(name, seed * 17 + s) for s, name in enumerate(policies)]
    game.apply(actor, action)
    for _ in range(50000):
        if game.winner is not None or game.turn >= root.turn + 1000:
            break
        seat, bot = game.current, bots[game.current]
        game.apply(seat, bot.choose(Observation.for_policy(game, seat, bot), seat))
    return {'winner': game.winner, 'win': int(game.winner == actor) if game.winner is not None else None,
            'truncated': game.winner is None, 'turns': game.turn - root.turn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--boards', type=int, required=True)
    parser.add_argument('--states-per-board', type=int, default=4)
    parser.add_argument('--replicates', type=int, default=4)
    args = parser.parse_args()
    if min(args.boards, args.states_per_board, args.replicates) < 1:
        raise ValueError('positive collection sizes required')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'source': source_files(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'seed': args.seed, 'boards': args.boards, 'states_per_board': args.states_per_board,
                'replicates': args.replicates, 'max_actions': 4, 'process_cap': 1}
    plan_path = args.output / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != identity:
            raise ValueError('collection identity changed')
    else:
        with plan_path.open('x') as stream:
            json.dump(identity, stream, indent=2)
    for board_seed in range(args.seed, args.seed + args.boards):
        path = args.output / f'{board_seed}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] != identity or saved['label_truncations']:
                raise ValueError('existing board has changed source or incomplete labels')
            continue
        names = ['planner_robber'] * 4 if board_seed % 2 == 0 else ['planner_robber', 'tactical', 'builder', 'expander']
        shift = board_seed % 4
        names = names[shift:] + names[:shift]
        root = Game.new(names, board_seed)
        bots = [make_policy(name, board_seed * 17 + s) for s, name in enumerate(names)]
        rows = []
        started = time.perf_counter()
        for step in range(50000):
            if root.winner is not None or root.turn >= 1000 or len(rows) == args.states_per_board:
                break
            seat, bot = root.current, bots[root.current]
            view = Observation.for_policy(root, seat, bot)
            action = bot.choose(view, seat)
            if root.phase == 'actions' and root.turn >= 8 + 20 * len(rows) and seat == len(rows) % 4:
                planner = make_policy('planner_robber', board_seed * 1000 + step)
                baseline = planner.choose(view, seat)
                choices = shortlist(view, baseline, planner)
                if len(choices) > 1:
                    label_started = time.perf_counter()
                    seeds = [board_seed * 10000 + len(rows) * args.replicates + r for r in range(args.replicates)]
                    for choice in choices:
                        choice['rollouts'] = [continuation(root, choice['action'], seat, names, seed) for seed in seeds]
                        choice['mean_win'] = (sum(r['win'] for r in choice['rollouts']) / args.replicates
                                              if not any(r['truncated'] for r in choice['rollouts']) else None)
                    row = {'board_seed': board_seed, 'step': step, 'turn': root.turn, 'seat': seat,
                           'replicate_seeds': seeds, 'choices': choices,
                           'seconds': time.perf_counter() - label_started}
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != 'choices'}), flush=True)
            root.apply(seat, action)
        if source_files() != identity['source']:
            raise RuntimeError('producer changed during collection')
        truncations = sum(r['truncated'] for row in rows for c in row['choices'] for r in c['rollouts'])
        with path.open('x') as stream:
            json.dump({'identity': identity, 'board_seed': board_seed, 'policies': names,
                       'rows': rows, 'label_truncations': truncations,
                       'seconds': time.perf_counter() - started}, stream, allow_nan=False)
        if truncations:
            raise RuntimeError('incomplete continuation labels; preserve shard and stop')


if __name__ == '__main__':
    main()
