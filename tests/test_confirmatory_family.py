import copy
from pathlib import Path
import runpy

import pytest

from catan_ai.search_eval import summarize


evaluate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              'scripts/read_confirmatory_family.py'))['evaluate']


def family():
    entries, results = [], []
    for index, control in enumerate(('tactical', 'external')):
        config = {'candidate': 'planner_robber', 'control': control, 'stage': 'heldout',
                  'requires_dev_pass': True, 'requires_search': False,
                  'seeds_per_panel': 128, 'inference': 'paired_normal',
                  'panels': [{'name': str(i), 'seed': 1000 * (index * 2 + i),
                              'opponents': ['tactical'] * 3} for i in range(2)],
                  'gates': {'maximum_match_truncations': 0, 'minimum_paired_win_lift': 0,
                            'minimum_paired_lcb': 0, 'maximum_p95_decision_seconds': 2,
                            'maximum_step_seconds': 2, 'maximum_turn_seconds': 10}}
        games = []
        for panel in config['panels']:
            for seed in range(panel['seed'], panel['seed'] + 128):
                for seat in range(4):
                    game = {'winner': seat, 'truncated': False, 'search': {},
                            'decision_seconds': [[.1]] * 4, 'step_seconds': [[.1]] * 4,
                            'turn_timings': [{'owner': seat, 'complete': True, 'seconds': .1}],
                            'external': {}}
                    baseline = copy.deepcopy(game)
                    baseline['winner'] = (seat + 1) % 4
                    if control == 'external':
                        baseline['external'][str(seat)] = {
                            'identity': {'commit': 'ecf931181b9a65bb4116a2153fb78c16f1438e00'},
                            'diagnostics': [{'fallback': False, 'budget_failure': False,
                                             'seconds': .1, 'turn': 1, 'phase': 'actions'}]}
                    games.append({'panel': panel['name'], 'seed': seed, 'seat': seat,
                                  'candidate': game, 'baseline': baseline})
        entries.append({'config': config})
        results.append({'config': config, 'source': {'test': 'fixture'},
                        'games': games, 'summary': summarize(config, games)})
    return {'runs': entries, 'source': {'test': 'fixture'}, 'family_alpha': .05,
            'comparison_count': 4}, results


def test_joint_reader_recomputes_with_four_comparison_alpha():
    plan, results = family()
    result = evaluate(plan, results)
    assert result['pass']
    assert len(result['panels']) == 4
    assert all(row['alpha'] == .0125 for row in result['panels'])
    assert result['panels'][0]['simultaneous_lcb'] < results[0]['summary']['panels']['0']['paired_seed_lcb']


def test_missing_run_source_drift_and_changed_summary_are_rejected():
    plan, results = family()
    with pytest.raises(ValueError, match='every planned'):
        evaluate(plan, results[:1])
    changed = copy.deepcopy(results)
    changed[1]['source']['test'] = 'changed'
    with pytest.raises(ValueError, match='frozen'):
        evaluate(plan, changed)
    results[0]['summary']['pass'] = False
    with pytest.raises(ValueError, match='raw games'):
        evaluate(plan, results)


def test_external_fault_cannot_be_hidden_by_other_winning_panels():
    plan, results = family()
    result = results[1]
    result['games'][0]['baseline']['external']['0']['diagnostics'][0]['fallback'] = True
    result['summary'] = summarize(result['config'], result['games'])
    readout = evaluate(plan, results)
    assert not readout['pass']
    assert readout['panels'][2]['simultaneous_lcb'] is None


def test_overlapping_panels_and_missing_clock_gate_rejected():
    plan, results = family()
    plan['runs'][1]['config']['panels'][0]['seed'] = 0
    with pytest.raises(ValueError, match='distinct'):
        evaluate(plan, results)
    plan, results = family()
    del plan['runs'][0]['config']['gates']['maximum_step_seconds']
    with pytest.raises(ValueError, match='time gates'):
        evaluate(plan, results)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1, True])
def test_invalid_clock_cannot_pass_via_comparison_behavior(invalid):
    plan, results = family()
    results[0]['games'][0]['candidate']['step_seconds'][0][0] = invalid
    with pytest.raises(ValueError, match='finite nonnegative'):
        evaluate(plan, results)


@pytest.mark.parametrize('defect', ['none', 'missing', 'wrong_sha', 'fallback', 'extra_seat'])
def test_learned_family_checks_every_game_checkpoint(defect):
    plan, results = family()
    checkpoint = {'sha256': 'a' * 64, 'kind': 'economic-parameters', 'path': 'fixture'}
    plan['checkpoint'] = checkpoint
    for entry, result in zip(plan['runs'], results):
        entry['config']['candidate'] = 'planner_parameter'
        result['checkpoint'] = checkpoint
        for row in result['games']:
            row['candidate']['policy_metadata'] = {str(row['seat']): {
                'model_sha256': checkpoint['sha256'], 'load_error': None}}
    metadata = results[0]['games'][0]['candidate']['policy_metadata']
    if defect == 'none':
        assert evaluate(plan, results)['pass']
        return
    if defect == 'missing':
        metadata.clear()
    elif defect == 'wrong_sha':
        metadata['0']['model_sha256'] = 'b' * 64
    elif defect == 'fallback':
        metadata['0']['load_error'] = 'unavailable'
    else:
        metadata['1'] = copy.deepcopy(metadata['0'])
    with pytest.raises(ValueError, match='per-game'):
        evaluate(plan, results)


def test_terminal_cli_reads_frozen_family_and_refuses_overwrite(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    from catan_ai.experiments import source_files

    plan, results = family()
    plan['source'] = source_files()
    for i, (entry, result) in enumerate(zip(plan['runs'], results, strict=True)):
        result['source'] = plan['source']
        path = tmp_path / f'run-{i}.json'
        path.write_text(json.dumps(result))
        entry['result'] = str(path)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / 'readout.json'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                                  'scripts/read_confirmatory_family.py'),
               '--plan', str(plan_path), '--output', str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    result = json.loads(output.read_text())
    assert result['pass']
    assert result['result_sha256'] == [hashlib.sha256(Path(e['result']).read_bytes()).hexdigest()
                                       for e in plan['runs']]
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert json.loads(output.read_text()) == result
