"""Bounded resumable policy-weight fitting; training boards only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

from catan_ai.experiments import source_files
from catan_ai.parameter_planner import DEFAULT, BOUNDS
from catan_ai.sim import play_game


def game_task(model, panel, seed, seat):
    os.environ['CATAN_PARAMETER_MODEL'] = model
    names = ['tactical'] * 3 if panel == 'tactical' else ['builder', 'tactical', 'expander']
    names.insert(seat, 'planner_parameter')
    return {'panel': panel, 'seed': seed, 'seat': seat, 'game': play_game(names, seed, 1000).as_dict()}


def summarize(rows, sha):
    wins = []
    if len(rows) != 64:
        raise ValueError('incomplete candidate training games')
    valid = True
    for panel, start in [('tactical', 1477000), ('mixed', 1477100)]:
        selected = [r for r in rows if r['panel'] == panel]
        if len(selected) != 32 or {(r['seed'], r['seat']) for r in selected} != {
                (seed, seat) for seed in range(start, start + 8) for seat in range(4)}:
            raise ValueError('missing or duplicate training seats')
        for row in selected:
            game = row['game']
            if game['truncated'] != (game['winner'] is None) or game['winner'] not in (None, 0, 1, 2, 3):
                raise ValueError('inconsistent training outcome')
            if game['policy_metadata'][str(row['seat'])] != {'model_sha256': sha, 'load_error': None}:
                raise ValueError('training checkpoint identity differs')
            valid &= not game['truncated'] and not game['external']
            valid &= max(t for samples in game['step_seconds'] for t in samples) <= 2
            turns = [t['seconds'] for t in game['turn_timings'] if t['complete']]
            valid &= bool(turns) and max(turns, default=math.inf) <= 10
        wins.append(sum(r['game']['winner'] == r['seat'] for r in selected) / 32)
    return {'panel_win_rates': wins, 'valid': bool(valid), 'fitness': [min(wins), sum(wins) / 2]}


def evaluate(root, trial, weights, identity, executor):
    directory = root / f'trial-{trial:03d}'
    directory.mkdir(exist_ok=True)
    model = directory / 'model.json'
    raw = json.dumps({'schema': 1, 'weights': weights, 'source': identity['source']}, indent=2).encode()
    if model.exists():
        if model.read_bytes() != raw:
            raise ValueError('resumed parameter proposal differs')
    else:
        with model.open('xb') as stream:
            stream.write(raw)
    sha = hashlib.sha256(raw).hexdigest()
    started = time.perf_counter()
    rows, futures = [], {}
    for panel, start in [('tactical', 1477000), ('mixed', 1477100)]:
        for seed in range(start, start + 8):
            for seat in range(4):
                path = directory / f'{seed}-{seat}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved['identity'] != identity or saved['model_sha256'] != sha:
                        raise ValueError('resumed training identity mismatch')
                    rows.append(saved['row'])
                else:
                    futures[executor.submit(game_task, str(model.resolve()), panel, seed, seat)] = path
    for future in as_completed(futures):
        row = future.result()
        if source_files() != identity['source']:
            raise RuntimeError('source changed during training')
        with futures[future].open('x') as stream:
            json.dump({'identity': identity, 'model_sha256': sha, 'row': row}, stream)
        rows.append(row)
    result = {**summarize(rows, sha), 'trial': trial, 'weights': weights, 'model_sha256': sha,
              'seconds_this_execution': time.perf_counter() - started, 'new_games': len(futures)}
    output = directory / 'summary.json'
    if output.exists():
        previous = json.loads(output.read_text())
        if any(previous[k] != result[k] for k in ('fitness', 'valid', 'panel_win_rates', 'model_sha256')):
            raise ValueError('stored training summary differs from games')
        result = previous
    else:
        with output.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    return result


def rank(result):
    return (result['valid'], *result['fitness'], -result['trial'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('capacity', 'fit'), required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    identity = {'source': source_files(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'host': platform.node(), 'process_cap': 2, 'seed': 1476999,
                'generations': 3, 'population': 8, 'elites': 3, 'bounds': BOUNDS}
    identity = json.loads(json.dumps(identity))
    plan = root / 'plan.json'
    if plan.exists():
        if json.loads(plan.read_text()) != identity:
            raise ValueError('training plan changed')
    else:
        with plan.open('x') as stream:
            json.dump(identity, stream, indent=2)
    if (root / 'model.json').exists():
        raise FileExistsError('selected checkpoint already frozen')
    with ProcessPoolExecutor(max_workers=2) as executor:
        best = evaluate(root, 0, DEFAULT, identity, executor)
        if not best['valid'] or best['new_games'] != 64 or best['seconds_this_execution'] > 120:
            raise RuntimeError('capacity bar failed; preserve data and stop')
        if args.mode == 'capacity':
            return
        rng = random.Random(1476999)
        means, scales = [math.log(x) for x in DEFAULT], [.5] * len(DEFAULT)
        all_trials = [best]
        for generation in range(3):
            population = [best]
            for proposal in range(7):
                weights = [min(hi, max(lo, math.exp(rng.gauss(mean, scale))))
                           for mean, scale, (lo, hi) in zip(means, scales, BOUNDS)]
                duplicate = next((r for r in all_trials if r['weights'] == weights), None)
                result = duplicate or evaluate(root, 1 + generation * 7 + proposal, weights, identity, executor)
                if duplicate is None:
                    all_trials.append(result)
                population.append(result)
            elite = sorted(population, key=rank, reverse=True)[:3]
            if not all(r['valid'] for r in elite):
                raise RuntimeError('too few valid elite policies')
            best = elite[0]
            logs = [[math.log(x) for x in r['weights']] for r in elite]
            means = [sum(r[i] for r in logs) / 3 for i in range(len(DEFAULT))]
            scales = [min(.7, max(.15, math.sqrt(sum((r[i] - means[i]) ** 2 for r in logs) / 3)))
                      for i in range(len(DEFAULT))]
        chosen = (root / f"trial-{best['trial']:03d}" / 'model.json').read_bytes()
        with (root / 'model.json').open('xb') as stream:
            stream.write(chosen)
        with (root / 'fit.json').open('x') as stream:
            json.dump({'identity': identity, 'selected': best, 'trials': all_trials}, stream, indent=2)
        print(json.dumps({'frozen_sha256': hashlib.sha256(chosen).hexdigest(), 'selected_trial': best['trial']}))


if __name__ == '__main__':
    main()
