"""DEV component ablation; all arms retain the same post-setup policy."""
import json
from pathlib import Path
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


def play(seed, seat, opponents, component):
    names = list(opponents)
    names.insert(seat, 'planner_robber')
    bots = [make_policy(name, seed * 17 + s + 1) for s, name in enumerate(names)]
    opening = make_policy('planner_opening', seed * 17 + seat + 1)
    game = Game.new(names, seed)
    first_city = first_expansion = None
    initial = None
    maximum_step = 0
    steps = 0
    while game.winner is None and game.turn < 1000 and steps < 50000:
        actor = game.current
        policy = bots[actor]
        if actor == seat and (component == 'both' and game.phase.startswith('setup')
                              or component == 'settlements' and game.phase == 'setup_settlement'
                              or component == 'roads' and game.phase == 'setup_road'):
            policy = opening
        started = time.perf_counter()
        view = Observation.for_policy(game, actor, policy)
        action = policy.choose(view, actor)
        was_setup = game.phase.startswith('setup')
        game.apply(actor, action)
        elapsed = time.perf_counter() - started
        if actor == seat:
            maximum_step = max(maximum_step, elapsed)
            if not was_setup and action['type'] == 'build_city' and first_city is None:
                first_city = game.turn
            if not was_setup and action['type'] == 'build_settlement' and first_expansion is None:
                first_expansion = game.turn
        if was_setup and not game.phase.startswith('setup'):
            p = game.players[seat]
            initial = {'settlements': sorted(p.settlements), 'roads': sorted(p.roads),
                       'hand': {r.value: n for r, n in p.resources.items()},
                       'income': {r.value: n for r, n in opening._income(game, seat).items()},
                       'value': opening._value(game, seat)}
        steps += 1
    return {'winner': game.winner, 'truncated': game.winner is None,
            'turns': game.turn, 'first_city': first_city, 'first_expansion': first_expansion,
            'setup': initial, 'maximum_step': maximum_step}


def main():
    output = Path('runs/opening-components-v1.jsonl')
    source = source_files()
    with output.open('x') as stream:
        stream.write(json.dumps({'source': source, 'seeds_per_panel': 4,
                                 'arms': ['baseline', 'settlements', 'roads', 'both']}) + '\n')
        for panel, start, opponents in (
            ('tactical', 1429000, ['tactical'] * 3),
            ('mixed', 1429100, ['builder', 'expander', 'tactical']),
        ):
            for seed in range(start, start + 4):
                for seat in range(4):
                    row = {'panel': panel, 'seed': seed, 'seat': seat}
                    for arm in ('baseline', 'settlements', 'roads', 'both'):
                        row[arm] = play(seed, seat, opponents, arm)
                    if source_files() != source:
                        raise RuntimeError('source changed during component probe')
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    print(json.dumps({'panel': panel, 'seed': seed, 'seat': seat}), flush=True)


if __name__ == '__main__':
    main()
