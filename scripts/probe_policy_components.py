"""Common-board opening/turn factorial; diagnostics cannot admit held-outs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform

from catan_ai.experiments import source_files
from catan_ai.sim import play_game


ARMS = ['tactical', 'planner_robber', 'planner_tactical_opening',
        'tactical_planner_opening', 'planner_external_opening', 'tactical_external_opening']
NEW = ['planner_tactical_opening', 'tactical_planner_opening', 'tactical_external_opening']


def task(panel, seed, seat, opponents):
    games = {}
    for arm in ARMS:
        names = list(opponents)
        names.insert(seat, arm)
        games[arm] = play_game(names, seed, 1000).as_dict()
    return {'panel': panel, 'seed': seed, 'seat': seat, 'games': games}


def summarize(rows):
    report = {}
    if len(rows) != 64 or {r['panel'] for r in rows} != {'tactical', 'mixed'}:
        raise ValueError('incomplete component experiment')
    identities = set()
    for panel in ('tactical', 'mixed'):
        selected = [r for r in rows if r['panel'] == panel]
        start = 1473000 if panel == 'tactical' else 1473100
        if len(selected) != 32 or {(r['seed'], r['seat']) for r in selected} != {
                (seed, seat) for seed in range(start, start + 8) for seat in range(4)}:
            raise ValueError('incomplete component panel')
        report[panel] = {}
        for arm in ARMS:
            games = [r['games'][arm] for r in selected]
            wins = sum(g['winner'] == r['seat'] for r, g in zip(selected, games))
            lift = sum((r['games'][arm]['winner'] == r['seat']) -
                       (r['games']['tactical']['winner'] == r['seat']) for r in selected) / 32
            faults = 0
            for row, game in zip(selected, games):
                expected = {str(row['seat'])} if 'external_opening' in arm else set()
                if set(game['external']) != expected:
                    raise ValueError('external evidence missing')
                for evidence in game['external'].values():
                    identities.add(json.dumps(evidence['identity'], sort_keys=True))
                    if evidence['identity']['commit'] != 'ecf931181b9a65bb4116a2153fb78c16f1438e00':
                        raise ValueError('external identity mismatch')
                    diagnostics = evidence['diagnostics']
                    if [d['phase'] for d in diagnostics] != ['setup_settlement', 'setup_road'] * 2:
                        raise ValueError('opening coverage mismatch')
                    faults += sum(d['fallback'] or d['budget_failure'] for d in diagnostics)
                if game['winner'] not in (None, 0, 1, 2, 3) or game['truncated'] != (game['winner'] is None):
                    raise ValueError('inconsistent outcome')
            maximum_step = max(t for g in games for times in g['step_seconds'] for t in times)
            maximum_turn = max(t['seconds'] for g in games for t in g['turn_timings'] if t['complete'])
            truncations = sum(g['truncated'] for g in games)
            report[panel][arm] = {'wins': wins, 'games': 32, 'lift': lift,
                                  'faults': faults, 'truncations': truncations,
                                  'maximum_step': maximum_step, 'maximum_turn': maximum_turn,
                                  'valid': faults == truncations == 0 and maximum_step <= 2 and maximum_turn <= 10}
    if len(identities) != 1:
        raise ValueError('external identity drift')
    eligible = [arm for arm in NEW if all(report[p][arm]['valid'] and report[p]['tactical']['valid']
                                        and report[p][arm]['lift'] >= .05 for p in report)]
    eligible.sort(key=lambda arm: (-min(report[p][arm]['lift'] for p in report),
                                   -sum(report[p][arm]['lift'] for p in report), arm))
    return {'panels': report, 'selected_for_fresh_dev': eligible[0] if eligible else None,
            'stage': 'diagnostic', 'pass_for_promotion': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'source': source_files(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'host': platform.node(), 'process_cap': 2, 'arms': ARMS}
    jobs, rows = [], []
    for panel, start, opponents in [('tactical', 1473000, ['tactical'] * 3),
                                     ('mixed', 1473100, ['builder', 'tactical', 'expander'])]:
        for seed in range(start, start + 8):
            for seat in range(4):
                path = args.output / f'{seed}-{seat}.json'
                if path.exists():
                    record = json.loads(path.read_text())
                    if record['identity'] != identity:
                        raise ValueError('existing shard identity mismatch')
                    rows.append(record['row'])
                else:
                    jobs.append((path, (panel, seed, seat, opponents)))
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(task, *job): path for path, job in jobs}
        for future in as_completed(futures):
            row = future.result()
            if source_files() != identity['source']:
                raise RuntimeError('source changed during diagnostic')
            with futures[future].open('x') as stream:
                json.dump({'identity': identity, 'row': row}, stream)
            rows.append(row)
            print(json.dumps({'completed': len(rows), 'total': 64}), flush=True)
    result = summarize(rows)
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
