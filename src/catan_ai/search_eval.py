"""Seat-rotated paired search screens with incremental, resumable game records."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from .experiments import source_files
from .sim import play_game
from .inference import paired_normal_lcb, planned_seeds
from .power import check_design, cluster_sigma, minimum_detectable_effect


def model_identity(config):
    policies = [config.get('candidate', 'mc'), config.get('control', 'heuristic')]
    policies.extend(n for panel in config['panels'] for n in panel['opponents'])
    if 'planner_parameter' in policies:
        if {'learned', 'planner_value', 'planner_value_draw'}.intersection(policies):
            raise ValueError('mixed learned checkpoint families require separate evaluations')
        from .parameter_planner import load_parameters
        _, sha, path = load_parameters()
        return {'path': str(path), 'sha256': sha, 'kind': 'economic-parameters'}
    if {'planner_value', 'planner_value_draw'}.intersection(policies):
        if 'learned' in policies:
            raise ValueError('mixed learned checkpoint families require separate evaluations')
        from .learned_planner import load_value_model, MODEL_PATH, MODEL_SHA
        load_value_model()
        return {'path': str(MODEL_PATH), 'sha256': MODEL_SHA, 'kind': 'outcome-value'}
    if 'learned' not in policies:
        return None
    from .learning import load_model
    path = os.environ.get('CATAN_MODEL_PATH')
    if not path:
        raise ValueError('learned evaluation requires CATAN_MODEL_PATH')
    _, sha = load_model(path)
    return {'path': str(Path(path).resolve()), 'sha256': sha}


def summarize(config, rows):
    panels = {}
    gates = config['gates']
    for panel in config['panels']:
        selected = [r for r in rows if r['panel'] == panel['name']]
        expected = {(panel['seed'] + offset, seat) for offset in range(config['seeds_per_panel']) for seat in range(4)}
        if len(selected) != len(expected) or {(r['seed'], r['seat']) for r in selected} != expected:
            raise ValueError('missing or duplicate seat/seed pairs')
        diagnostics = [d for r in selected for d in r['candidate']['search'].get(str(r['seat']), [])]
        requires_search = config.get('requires_search', True)
        reasons = Counter(d['reason'] for d in diagnostics)
        attempted = [d for d in diagnostics if d['reason'] not in {'unsupported_phase', 'single_candidate'}]
        coverage = sum(d['rollouts'] > 0 for d in attempted) / len(attempted) if attempted else 0
        times = sorted(d['seconds'] for d in attempted)
        if not requires_search:
            times = sorted(t for r in selected for t in r['candidate']['decision_seconds'][r['seat']])
        p95 = times[max(0, math.ceil(.95 * len(times)) - 1)] if times else math.inf
        rollouts = sum(d['rollouts'] for d in diagnostics)
        rollout_truncation = sum(d['truncated'] for d in diagnostics) / rollouts if rollouts else 1
        truncated = sum(r[arm]['truncated'] for r in selected for arm in ('candidate', 'baseline'))
        external_faults = 0
        external_identities = set()
        for r in selected:
            for arm in ('candidate', 'baseline'):
                result = r[arm]
                if result['truncated'] != (result['winner'] is None) or result['winner'] not in (None, 0, 1, 2, 3):
                    raise ValueError('invalid terminal outcome')
                names = list(panel['opponents'])
                names.insert(r['seat'], config.get('candidate', 'mc') if arm == 'candidate'
                             else config.get('control', 'heuristic'))
                expected_external = {str(s) for s, name in enumerate(names)
                                     if name in ('external', 'planner_external_opening', 'tactical_external_opening')}
                external = result.get('external', {})
                if set(external) != expected_external:
                    raise ValueError('external comparator evidence missing or unexpected')
                for seat, evidence in external.items():
                    identity = evidence.get('identity')
                    records = evidence.get('diagnostics', [])
                    if not identity or not records:
                        raise ValueError('external comparator identity or decisions missing')
                    external_identities.add(json.dumps(identity, sort_keys=True))
                    if identity['commit'] != 'ecf931181b9a65bb4116a2153fb78c16f1438e00':
                        raise ValueError('external comparator revision mismatch')
                    step_times = result.get('step_seconds', [[], [], [], []])[int(seat)]
                    opening_only = names[int(seat)] in ('planner_external_opening', 'tactical_external_opening')
                    if opening_only and ([d['phase'] for d in records] !=
                                         ['setup_settlement', 'setup_road'] * 2 or len(step_times) < 4):
                        raise ValueError('external opening coverage incomplete')
                    if not opening_only and len(step_times) != len(records):
                        raise ValueError('external timing coverage incomplete')
                    external_faults += sum(bool(d['fallback'] or d['budget_failure']) for d in records)
                    external_faults += sum(t > 2 for t in step_times)
                    turn_compute = {}
                    for d in records:
                        if not d['phase'].startswith('setup'):
                            turn_compute[d['turn']] = turn_compute.get(d['turn'], 0) + d['seconds']
                    external_faults += sum(t > 10 for t in turn_compute.values())
        if len(external_identities) > 1:
            raise ValueError('external comparator package identity drift')
        differences = [int(r['candidate']['winner'] == r['seat']) - int(r['baseline']['winner'] == r['seat']) for r in selected]
        lift = sum(differences) / len(differences) if not truncated and not external_faults else None
        seeds = sorted({r['seed'] for r in selected})
        clusters = [sum(d for r, d in zip(selected, differences) if r['seed'] == seed) / 4 for seed in seeds]
        # Each seed's four-seat paired average lies in [-1, 1].
        # Bonferroni covers the two predeclared opponent panels.
        lcb = max(-1, lift - math.sqrt(2 * math.log(len(config['panels']) / .05)
                                      / config['seeds_per_panel'])) if lift is not None else None
        method = config.get('inference', 'hoeffding')
        if method == 'paired_normal':
            lcb = paired_normal_lcb(clusters, .05 / len(config['panels'])) if lift is not None else None
        elif method != 'hoeffding':
            raise ValueError('unknown inference method')
        passed = (truncated <= gates['maximum_match_truncations'] and lift is not None
                  and lift >= gates['minimum_paired_win_lift']
                  and (not requires_search or coverage >= gates['minimum_search_coverage'])
                  and p95 <= gates['maximum_p95_decision_seconds']
                  and (not requires_search or rollout_truncation <= gates['maximum_rollout_truncation_rate']))
        if 'minimum_paired_lcb' in gates:
            passed = passed and lcb is not None and lcb > gates['minimum_paired_lcb']
        panels[panel['name']] = {'pairs': len(selected), 'paired_win_lift': lift,
                               'paired_seed_lcb': lcb, 'search_coverage': coverage,
                               'p95_seconds': p95, 'rollout_truncation_rate': rollout_truncation,
                               'match_truncations': truncated, 'reasons': dict(reasons),
                               'external_faults': external_faults,
                               'rollouts': rollouts, 'pass': passed}
        panels[panel['name']].update(inference=method, seed_cluster_differences=clusters,
                                    suggested_heldout_seeds=planned_seeds(clusters, alpha=.05 / len(config['panels'])) if len(clusters) > 1 and lift is not None else None)
        if len(clusters) > 1:
            # Descriptive only: how small a true lift this panel's design could have seen.
            sigma = cluster_sigma(clusters)
            panels[panel['name']]['cluster_sd'] = sigma
            panels[panel['name']]['minimum_detectable_effect_80'] = minimum_detectable_effect(
                sigma, len(clusters), .05 / len(config['panels']))
        for arm in ('candidate', 'baseline'):
            steps = sorted(t for r in selected for t in r[arm].get('step_seconds', [[], [], [], []])[r['seat']])
            turns = sorted(t['seconds'] for r in selected for t in r[arm].get('turn_timings', [])
                           if t['owner'] == r['seat'] and t['complete'])
            panels[panel['name']][arm + '_latency'] = {
                'step_count': len(steps), 'turn_count': len(turns),
                'step_p95_seconds': steps[math.ceil(.95 * len(steps)) - 1] if steps else None,
                'turn_p95_seconds': turns[math.ceil(.95 * len(turns)) - 1] if turns else None,
                'step_max_seconds': max(steps) if steps else None,
                'turn_max_seconds': max(turns) if turns else None,
            }
            for key, samples in (('maximum_step_seconds', steps), ('maximum_turn_seconds', turns)):
                if key in gates and (not samples or max(samples) > gates[key]):
                    panels[panel['name']]['pass'] = False
    result = {'panels': panels, 'pass': all(p['pass'] for p in panels.values())}
    if config.get('stage') == 'regression_audit':
        for panel in panels.values():
            clusters = panel['seed_cluster_differences']
            panel['regression_ucb'] = (-paired_normal_lcb([-x for x in clusters], .05 / len(panels))
                                       if panel['match_truncations'] == 0 and panel['external_faults'] == 0 else None)
        result.update(pass_for_promotion=False, purpose='non-promotion regression audit', **{'pass': False})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dev-result', type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = source_files()
    checkpoint = model_identity(config)
    audit = config.get('stage') == 'regression_audit'
    check_design(config)
    if audit and (config.get('inference') != 'paired_normal' or config['seeds_per_panel'] < 128):
        raise ValueError('regression audit requires calibrated seed-cluster inference and >=128 seeds')
    if config.get('requires_dev_pass') or audit:
        if not args.dev_result:
            raise ValueError('held-out evaluation requires a passing DEV result')
        dev = json.loads(args.dev_result.read_text())
        if dev.get('checkpoint') != checkpoint:
            raise ValueError('DEV must evaluate this checkpoint')
        same_source = dev['source'] == source
        if audit:
            # Audit reporting may change; gameplay and feature producers must stay frozen.
            same_source = ({k: v for k, v in dev['source'].items() if k != 'search_eval.py'} ==
                           {k: v for k, v in source.items() if k != 'search_eval.py'})
        if ((not audit and not dev['summary']['pass']) or (audit and dev['summary']['pass'])
                or not same_source or dev['config']['stage'] != 'dev'):
            raise ValueError('DEV must pass on this source before held-out evaluation')
        if dev['config'].get('candidate', 'mc') != config.get('candidate', 'mc'):
            raise ValueError('DEV must evaluate this candidate')
        if dev['config'].get('control', 'heuristic') != config.get('control', 'heuristic'):
            raise ValueError('DEV must evaluate this control')
        if [p['opponents'] for p in dev['config']['panels']] != [p['opponents'] for p in config['panels']]:
            raise ValueError('DEV must evaluate these opponent lineups')
        used = {p['seed'] + i for p in dev['config']['panels'] for i in range(dev['config']['seeds_per_panel'])}
        reserved = {p['seed'] + i for p in config['panels'] for i in range(config['seeds_per_panel'])}
        if used & reserved:
            raise ValueError('held-out seeds overlap DEV')
        if config.get('inference') == 'paired_normal':
            if not args.calibration:
                raise ValueError('approximate inference requires its calibration result')
            calibration = json.loads(args.calibration.read_text())
            if not calibration['pass'] or calibration['source']['inference.py'] != source['inference.py']:
                raise ValueError('inference calibration does not match this implementation')
            if calibration['alpha'] != .05 / len(config['panels']):
                raise ValueError('inference calibration must match per-panel alpha')
    identity = {'config': config, 'source': source, 'host': platform.node(), 'process_cap': 1}
    if checkpoint is not None:
        identity['checkpoint'] = checkpoint
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix('.jsonl')
    rows = []
    if journal.exists():
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        if not entries or entries[0] != {'identity': identity, 'signature': signature}:
            raise ValueError('journal belongs to a different source/config/host')
        rows = entries[1:]
    else:
        with journal.open('x') as stream:
            stream.write(json.dumps({'identity': identity, 'signature': signature}) + '\n')
    if args.output.exists():
        raise ValueError('completed output already exists')
    done = {(r['panel'], r['seed'], r['seat']) for r in rows}
    for panel in config['panels']:
        for offset in range(config['seeds_per_panel']):
            seed = panel['seed'] + offset
            for seat in range(4):
                if (panel['name'], seed, seat) in done:
                    continue
                if source_files() != source or model_identity(config) != checkpoint:
                    raise RuntimeError('source changed during evaluation')
                started = time.monotonic()
                policies = list(panel['opponents'])
                policies.insert(seat, config.get('candidate', 'mc'))
                candidate = play_game(policies, seed, config['max_turns'])
                policies[seat] = config.get('control', 'heuristic')
                baseline = play_game(policies, seed, config['max_turns'])
                row = {'panel': panel['name'], 'seed': seed, 'seat': seat,
                       'candidate': candidate.as_dict(), 'baseline': baseline.as_dict(),
                       'seconds': time.monotonic() - started}
                if source_files() != source or model_identity(config) != checkpoint:
                    raise RuntimeError('source changed during evaluation')
                with journal.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                rows.append(row)
                print(json.dumps({'panel': panel['name'], 'seed': seed, 'seat': seat,
                                  'seconds': row['seconds']}), flush=True)
    result = {**identity, 'summary': summarize(config, rows), 'games': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
