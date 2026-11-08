"""Profile ``planner_pv2_puct`` decisions in full seeded games (no screen, no gate).

Plays ``--games`` games of the candidate against three ``--opponent`` bots (the
candidate's seat rotates with the game index) and reports, per searched decision:
latency (p50/p95/max), simulations (p50/p95/min), the deepest simulation (our turns),
the principal-variation depth distribution and the fallback count; per game step and
own-turn wall clock (the screen's step/turn gates). Output: one JSON document.

    PYTHONPATH=src uv run --with numpy python scripts/puct_profile.py --games 2 --seed 2950000
"""
import argparse
from collections import Counter
import json
import math
import platform
import subprocess
import time

from catan_ai.engine import BACKEND
from catan_ai.sim import play_game


def quantile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def summary(values):
    return {'n': len(values), 'p50': quantile(values, .5), 'p95': quantile(values, .95),
            'max': max(values) if values else None, 'min': min(values) if values else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2950000)
    parser.add_argument('--opponent', default='tactical')
    parser.add_argument('--candidate', default='planner_pv2_puct')
    parser.add_argument('--max-turns', type=int, default=1000)
    args = parser.parse_args()
    started = time.time()
    games, records, steps, turns, decisions = [], [], [], [], []
    for g in range(args.games):
        seat = g % 4
        names = [args.opponent] * 4
        names[seat] = args.candidate
        t0 = time.perf_counter()
        result = play_game(names, args.seed + g, args.max_turns)
        own = result.search.get(str(seat), [])
        records += own
        steps += list(result.step_seconds[seat])
        decisions += list(result.decision_seconds[seat])
        own_turns = [t['seconds'] for t in result.turn_timings if t['owner'] == seat and t['complete']]
        turns += own_turns
        searches_per_turn = Counter(d['turn'] for d in own if d['phase'] == 'actions')
        games.append({'seed': args.seed + g, 'seat': seat, 'winner': result.winner, 'turns': result.turns,
                      'truncated': result.truncated, 'seconds': time.perf_counter() - t0,
                      'searched_decisions': len(own), 'fallbacks': sum(d['fallback'] for d in own),
                      'max_searches_in_one_turn': max(searches_per_turn.values(), default=0),
                      'step_max_seconds': max(result.step_seconds[seat]),
                      'turn_max_seconds': max(own_turns, default=None)})
    by_phase = {}
    for phase in sorted({d['phase'] for d in records}):
        rows = [d for d in records if d['phase'] == phase]
        by_phase[phase] = {'latency_seconds': summary([d['seconds'] for d in rows]),
                           'simulations': summary([d['simulations'] for d in rows]),
                           'fallbacks': sum(d['fallback'] for d in rows)}
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        revision = None
    report = {
        'config': vars(args), 'revision': revision, 'engine': BACKEND, 'host': platform.node(),
        'elapsed_seconds': time.time() - started, 'games': games,
        'searched_decisions': len(records),
        'latency_seconds': summary([d['seconds'] for d in records]),
        'simulations': summary([d['simulations'] for d in records]),
        'max_depth_reached': max((d['max_depth'] for d in records), default=None),
        'max_depth_distribution': dict(sorted(Counter(d['max_depth'] for d in records).items())),
        'pv_depth_distribution': dict(sorted(Counter(d['pv_depth'] for d in records).items())),
        'pv_depth_mean': sum(d['pv_depth'] for d in records) / len(records) if records else None,
        'fallbacks': sum(d['fallback'] for d in records),
        'abandoned': sum(d['abandoned'] for d in records),
        'cut_passes': sum(d['truncated'] for d in records),
        'by_phase': by_phase,
        'all_decisions_seconds': summary(decisions),
        'step_seconds': summary(steps),
        'own_turn_seconds': summary(turns),
        'gates_seen': {'p95_decision_le_2s': (quantile(decisions, .95) or 0) <= 2,
                       'step_max_le_2s': (max(steps) if steps else 0) <= 2,
                       'turn_max_le_10s': (max(turns) if turns else 0) <= 10},
    }
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
