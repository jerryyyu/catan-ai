"""Measure search-budget saturation without changing the planner's decisions."""
import argparse
import json
from pathlib import Path
import random
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.planner import ProductionRobberPlanner


class ProfilePlanner(ProductionRobberPlanner):
    def _value(self, game, seat):
        self.value_calls = getattr(self, 'value_calls', 0) + 1
        return super()._value(game, seat)


def eligible(action):
    return action['type'] in ('build_city', 'build_settlement', 'build_road',
                              'bank_trade', 'buy_development') or (
        action['type'] == 'play_development' and
        action['card'] in ('year_of_plenty', 'road_building'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_files()
    plan = {'source': source, 'seeds': list(range(1460000, 1460004)),
            'seats': list(range(4)), 'opponents': ['tactical'] * 3,
            'minimum_saturation_fraction': .10, 'maximum_truncations': 0,
            'require_identical_actions_and_rng': True}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    games = []
    for seed in plan['seeds']:
        for focal in plan['seats']:
            names = ['tactical'] * 4
            names[focal] = 'planner_robber'
            game = Game.new(names, seed)
            policies = [make_policy(name, seed * 17 + seat + 1) for seat, name in enumerate(names)]
            policies[focal] = ProfilePlanner(random.Random(seed * 17 + focal + 1))
            records = []
            started = time.perf_counter()
            steps = 0
            while game.winner is None and game.turn < 1000 and steps < 50000:
                seat, policy = game.current, policies[game.current]
                view = Observation.for_policy(game, seat, policy)
                if seat == focal:
                    reference = ProductionRobberPlanner(random.Random(0))
                    reference.rng.setstate(policy.rng.getstate())
                    policy.value_calls = 0
                    action = policy.choose(view, seat)
                    expected = reference.choose(view, seat)
                    if action != expected or policy.rng.getstate() != reference.rng.getstate():
                        raise RuntimeError('instrumentation changed decisions or RNG')
                    if view.phase in ('actions', 'free_roads'):
                        count = sum(eligible(a) for a in view.actions)
                        records.append({'turn': view.turn, 'phase': view.phase,
                                        'root_candidates': count,
                                        'nodes': policy.value_calls - 1,
                                        'chosen_type': action['type']})
                else:
                    action = policy.choose(view, seat)
                game.apply(seat, action)
                steps += 1
            row = {'seed': seed, 'seat': focal, 'winner': game.winner,
                   'truncated': game.winner is None, 'records': records,
                   'seconds': time.perf_counter() - started}
            games.append(row)
            with (args.output / 'games.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k != 'records'}), flush=True)
            if source_files() != source:
                raise RuntimeError('producer changed during profile')
    decisions = [r for g in games for r in g['records'] if r['root_candidates']]
    saturated = sum(r['nodes'] == 128 for r in decisions)
    fraction = saturated / len(decisions) if decisions else 0
    summary = {'games': len(games), 'decisions_with_candidates': len(decisions),
               'saturated': saturated, 'saturation_fraction': fraction,
               'root_omissions': sum(r['root_candidates'] > 128 for r in decisions),
               'max_nodes': max((r['nodes'] for r in decisions), default=0),
               'truncations': sum(g['truncated'] for g in games),
               'pass': fraction >= .10 and not any(g['truncated'] for g in games)}
    (args.output / 'summary.json').write_text(json.dumps({'plan': plan, 'summary': summary}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
