"""Compare separated offline event streams on preserved training positions."""
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
from diagnose_paired_labels import summarize
from train_paired_value import read_dataset


class EventStreams:
    def __init__(self, seed):
        self.dice = random.Random(seed)
        self.theft = random.Random(seed ^ 0xB41F82)

    def randint(self, lo, hi):
        if (lo, hi) != (1, 6):
            raise ValueError('unexpected dice request')
        return self.dice.randint(lo, hi)

    def choice(self, cards):
        return self.theft.choice(cards)


def continuation(root, action, actor, names, seed):
    game = deepcopy(root, {id(root.board): root.board})
    game.rng = EventStreams(seed)
    random.Random(seed ^ 0x917ABC).shuffle(game.deck)
    game.record_events, game.events = False, []
    policies = list(names)
    policies[actor] = 'planner_robber'
    bots = [make_policy(name, seed * 17 + seat) for seat, name in enumerate(policies)]
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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, producer, hashes = read_dataset(args.root / 'train', 1464000, 64)
    if source_files() != producer['source']:
        raise ValueError('cannot reconstruct with changed package source')
    identity = {'producer': producer, 'training_hashes': hashes,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    changed, controls = [], []
    for seed in range(1464000, 1464008):
        selected = [row for row in rows if row['board_seed'] == seed]
        path = args.output / f'{seed}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] != identity:
                raise ValueError('diagnostic identity changed')
            changed.extend(saved['rows'])
            controls.extend(selected)
            continue
        board = json.loads((args.root / 'train' / f'{seed}.json').read_text())
        names = board['policies']
        game = Game.new(names, seed)
        bots = [make_policy(name, seed * 17 + seat) for seat, name in enumerate(names)]
        targets = {row['step']: row for row in selected}
        collected = []
        for step in range(max(targets) + 1):
            seat, bot = game.current, bots[game.current]
            view = Observation.for_policy(game, seat, bot)
            action = bot.choose(view, seat)
            if step in targets:
                original = targets[step]
                planner = make_policy('planner_robber', seed * 1000 + step)
                choices = shortlist(view, planner.choose(view, seat), planner)
                assert seat == original['seat'] and game.turn == original['turn']
                assert choices == [{k: v for k, v in c.items() if k not in ('rollouts', 'mean_win')}
                                   for c in original['choices']]
                started = time.perf_counter()
                for choice in choices:
                    choice['rollouts'] = [continuation(game, choice['action'], seat, names, s)
                                          for s in original['replicate_seeds']]
                    if any(r['truncated'] for r in choice['rollouts']):
                        raise RuntimeError('incomplete continuation')
                    choice['mean_win'] = sum(r['win'] for r in choice['rollouts']) / 4
                collected.append({**original, 'choices': choices, 'seconds': time.perf_counter() - started})
                print(json.dumps({'seed': seed, 'step': step, 'seconds': collected[-1]['seconds']}), flush=True)
            game.apply(seat, action)
        if source_files() != producer['source']:
            raise RuntimeError('source changed during diagnostic')
        with path.open('x') as stream:
            json.dump({'identity': identity, 'rows': collected}, stream, allow_nan=False)
        changed.extend(collected)
        controls.extend(selected)
    baseline, candidate = summarize(controls), summarize(changed)
    ratio = candidate['estimated_four_replicate_mean_noise'] / baseline['estimated_four_replicate_mean_noise']
    maximum = max(row['seconds'] for row in changed)
    result = {'control': baseline, 'candidate': candidate, 'noise_ratio': ratio,
              'maximum_position_seconds': maximum,
              'pass': len(changed) >= 24 and candidate['pass'] and ratio <= .8 and maximum <= 30}
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
