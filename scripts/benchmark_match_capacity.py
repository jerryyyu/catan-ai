"""Measure identical diagnostic games under bounded match concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from catan_ai.experiments import source_files
from catan_ai.sim import play_game
from catan_ai.search_eval import model_identity


def cases(candidate='planner_robber', seed=1455000):
    result = []
    for seat in range(4):
        for focal in (candidate, 'external'):
            names = ['builder', 'external', 'expander']
            names.insert(seat, focal)
            result.append({'seed': seed + seat % 2, 'policies': names})
    return result


def validity(game, policies):
    steps = [t for seat in game['step_seconds'] for t in seat]
    turns = [t['seconds'] for t in game['turn_timings'] if t['complete']]
    compute = []
    diagnostic_clocks = []
    faults = 0
    coverage = True
    for seat, evidence in game['external'].items():
        records = evidence['diagnostics']
        coverage &= bool(evidence.get('identity') and records)
        coverage &= len(records) == len(game['step_seconds'][int(seat)])
        per_turn = {}
        for d in records:
            diagnostic_clocks.append(d['seconds'])
            faults += bool(d['fallback'] or d['budget_failure'])
            if not d['phase'].startswith('setup'):
                per_turn[d['turn']] = per_turn.get(d['turn'], 0) + d['seconds']
        compute.extend(per_turn.values())
    clocks = steps + turns + compute + diagnostic_clocks
    expected = {str(s) for s, name in enumerate(policies) if name == 'external'}
    valid = (coverage and set(game['external']) == expected
             and type(game['winner']) is int and game['winner'] in range(4)
             and len(game['step_seconds']) == 4
             and all(game['step_seconds'])
             and bool(steps and turns and compute)) and all(
        isinstance(t, (int, float)) and not isinstance(t, bool)
        and math.isfinite(t) and t >= 0 for t in clocks)
    return {'pass': bool(valid and not game['truncated'] and not faults
                         and max(steps) <= 2 and max(turns) <= 10
                         and max(compute) <= 10),
            'faults': faults, 'truncated': game['truncated'],
            'max_step': max(steps, default=None),
            'max_turn_wall': max(turns, default=None),
            'max_external_turn_compute': max(compute, default=None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--case', type=int, choices=range(8))
    parser.add_argument('--candidate', choices=('planner_robber', 'planner_parameter'), default='planner_robber')
    parser.add_argument('--seed', type=int, default=1455000)
    args = parser.parse_args()
    if platform.node() != args.host:
        raise ValueError('benchmark host differs from declared host')
    if args.case is not None:
        case = cases(args.candidate, args.seed)[args.case]
        game = play_game(case['policies'], case['seed'], 1000).as_dict()
        with args.output.open('x') as stream:
            json.dump({'case': case, 'game': game,
                       'validity': validity(game, case['policies'])}, stream)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_files()
    checkpoint_config = {'candidate': args.candidate, 'control': 'external', 'panels': []}
    checkpoint = model_identity(checkpoint_config)
    plan = {'host': args.host, 'source': source, 'cases': cases(args.candidate, args.seed),
            'checkpoint': checkpoint,
            'max_turns': 1000,
            'match_process_caps': [1, 2, 4], 'max_external_workers_per_match': 2,
            'minimum_speedup': 1.25, 'max_step_seconds': 2, 'max_turn_seconds': 10}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    waves = []
    reference_outcomes = None
    for cap in plan['match_process_caps']:
        if source_files() != source or model_identity(checkpoint_config) != checkpoint:
            raise RuntimeError('source changed during capacity benchmark')
        directory = args.output / str(cap)
        directory.mkdir()

        def run(index, directory=directory):
            destination = directory / f'{index}.json'
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--host', args.host, '--output', str(destination),
                            '--candidate', args.candidate, '--seed', str(args.seed),
                            '--case', str(index)], check=True)
            result = json.loads(destination.read_text())
            if checkpoint is not None:
                for seat, name in enumerate(result['case']['policies']):
                    if name == args.candidate and result['game']['policy_metadata'].get(str(seat)) != {
                            'model_sha256': checkpoint['sha256'], 'load_error': None}:
                        raise ValueError('capacity candidate checkpoint mismatch')
            return result

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=cap) as pool:
            results = list(pool.map(run, range(len(plan['cases']))))
        seconds = time.perf_counter() - started
        if source_files() != source or model_identity(checkpoint_config) != checkpoint:
            raise RuntimeError('source changed during capacity benchmark')
        wave = {'match_process_cap': cap, 'seconds': seconds,
                'validity': [r['validity'] for r in results]}
        outcomes = [(r['game']['winner'], r['game']['turns'], r['game']['points'])
                    for r in results]
        if reference_outcomes is None:
            reference_outcomes = outcomes
        wave['outcomes_changed_from_serial'] = sum(
            a != b for a, b in zip(reference_outcomes, outcomes, strict=True))
        wave['speedup'] = waves[0]['seconds'] / seconds if waves else 1.0
        wave['pass'] = all(r['pass'] for r in wave['validity']) and (
            cap == 1 or wave['speedup'] >= plan['minimum_speedup'])
        waves.append(wave)
        (directory / 'summary.json').write_text(json.dumps(wave, indent=2))
        print(json.dumps(wave), flush=True)
        if not wave['pass']:
            break
    (args.output / 'summary.json').write_text(json.dumps({'plan': plan, 'waves': waves}, indent=2))


if __name__ == '__main__':
    main()
