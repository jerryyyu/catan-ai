import copy

import pytest

from catan_ai.search_eval import summarize


def fixture():
    config = {'panels': [{'name': 'h', 'seed': 10, 'opponents': ['heuristic'] * 3}], 'seeds_per_panel': 1,
              'gates': {'maximum_match_truncations': 0, 'minimum_paired_win_lift': .05,
                        'minimum_search_coverage': .8, 'maximum_p95_decision_seconds': 2,
                        'maximum_rollout_truncation_rate': .01}}
    rows = []
    for seat in range(4):
        diagnostic = {'reason': 'confirmed', 'rollouts': 10, 'truncated': 0, 'seconds': .5}
        rows.append({'panel': 'h', 'seed': 10, 'seat': seat,
                     'candidate': {'winner': seat, 'truncated': False, 'search': {str(seat): [diagnostic]}},
                     'baseline': {'winner': (seat + 1) % 4, 'truncated': False}})
    return config, rows


def test_dev_signal_is_not_heldout_confidence():
    config, rows = fixture()
    assert summarize(config, rows)['pass']
    config['gates']['minimum_paired_lcb'] = 0
    assert not summarize(config, rows)['pass']


def test_optional_maximum_clocks_fail_closed_for_either_arm():
    config, rows = fixture()
    config['gates'].update(maximum_step_seconds=2, maximum_turn_seconds=10)
    assert not summarize(config, rows)['pass']
    for row in rows:
        for arm in ('candidate', 'baseline'):
            row[arm]['step_seconds'] = [[.1] for _ in range(4)]
            row[arm]['turn_timings'] = [{'owner': row['seat'], 'complete': True, 'seconds': 1}]
    assert summarize(config, rows)['pass']
    rows[0]['baseline']['step_seconds'][0].append(2.1)
    assert not summarize(config, rows)['pass']
    rows[0]['baseline']['step_seconds'][0].pop()
    rows[0]['candidate']['turn_timings'][0]['seconds'] = 11
    assert not summarize(config, rows)['pass']


def test_missing_duplicate_and_truncated_pairs_cannot_pass():
    config, rows = fixture()
    with pytest.raises(ValueError):
        summarize(config, rows[:-1])
    with pytest.raises(ValueError):
        summarize(config, rows[:-1] + [copy.deepcopy(rows[0])])
    rows[0]['candidate'].update(winner=None, truncated=True)
    result = summarize(config, rows)
    assert not result['pass']
    assert result['panels']['h']['paired_win_lift'] is None


def test_fast_fallbacks_cannot_hide_slow_or_missing_search():
    config, rows = fixture()
    for row in rows:
        ds = row['candidate']['search'][str(row['seat'])]
        ds[0]['seconds'] = 3
        ds.extend([{'reason': 'unsupported_phase', 'rollouts': 0, 'truncated': 0, 'seconds': .0001}] * 100)
    assert not summarize(config, rows)['pass']
    for row in rows:
        row['candidate']['search'][str(row['seat'])][0].update(reason='belief_failure', rollouts=0, seconds=.1)
    assert not summarize(config, rows)['pass']


def test_step_and_turn_latency_are_separate_from_policy_gate():
    config, rows = fixture()
    for row in rows:
        row['candidate']['step_seconds'] = [[3.0] for _ in range(4)]
        row['candidate']['turn_timings'] = [
            {'owner': row['seat'], 'complete': True, 'seconds': 10.0},
            {'owner': row['seat'], 'complete': False, 'seconds': 100.0},
            {'owner': (row['seat'] + 1) % 4, 'complete': True, 'seconds': 200.0},
        ]
    result = summarize(config, rows)
    assert result['pass']
    panel = result['panels']['h']
    assert panel['p95_seconds'] == .5
    assert panel['candidate_latency'] == {'step_count': 4, 'turn_count': 4,
                                          'step_p95_seconds': 3.0, 'turn_p95_seconds': 10.0,
                                          'step_max_seconds': 3.0, 'turn_max_seconds': 10.0}
    assert panel['baseline_latency']['turn_p95_seconds'] is None


def test_regression_audit_cannot_promote_even_on_positive_results():
    config, original = fixture()
    config.update(stage='regression_audit', inference='paired_normal', seeds_per_panel=128)
    rows = []
    for seed in range(10, 138):
        for row in original:
            item = copy.deepcopy(row)
            item['seed'] = seed
            rows.append(item)
    result = summarize(config, rows)
    assert result['panels']['h']['pass']
    assert not result['pass'] and not result['pass_for_promotion']
    for row in rows:
        row['candidate']['winner'], row['baseline']['winner'] = row['baseline']['winner'], row['candidate']['winner']
    result = summarize(config, rows)
    assert result['panels']['h']['regression_ucb'] < 0
    assert not result['pass']


def test_external_comparator_failures_withhold_strength_in_either_arm():
    config, rows = fixture()
    config['panels'][0]['opponents'] = ['external'] * 3
    identity = {'commit': 'ecf931181b9a65bb4116a2153fb78c16f1438e00', 'package_sha256': 'test'}
    for row in rows:
        for arm in ('candidate', 'baseline'):
            row[arm]['step_seconds'] = [[.1] for _ in range(4)]
            row[arm]['external'] = {
                str(s): {'identity': identity, 'diagnostics': [
                    {'fallback': False, 'budget_failure': False, 'seconds': .1, 'turn': 1, 'phase': 'actions'}]}
                for s in range(4) if s != row['seat']}
    assert summarize(config, rows)['pass']
    for arm in ('candidate', 'baseline'):
        failed = copy.deepcopy(rows)
        failed[0][arm]['external']['1']['diagnostics'][0]['fallback'] = True
        result = summarize(config, failed)
        assert not result['pass']
        assert result['panels']['h']['paired_win_lift'] is None
    del rows[0]['baseline']['external']['1']
    with pytest.raises(ValueError, match='evidence missing'):
        summarize(config, rows)


def test_external_opening_requires_all_four_setup_decisions():
    config, rows = fixture()
    config['candidate'] = 'planner_external_opening'
    for row in rows:
        seat = row['seat']
        row['candidate']['step_seconds'] = [[.1] * 7 for _ in range(4)]
        row['candidate']['external'] = {str(seat): {
            'identity': {'commit': 'ecf931181b9a65bb4116a2153fb78c16f1438e00'},
            'diagnostics': [{'phase': phase, 'turn': 0, 'seconds': .1,
                             'fallback': False, 'budget_failure': False}
                            for phase in ['setup_settlement', 'setup_road'] * 2]}}
    assert summarize(config, rows)['pass']
    failed = copy.deepcopy(rows)
    failed[0]['candidate']['external']['0']['diagnostics'][0]['fallback'] = True
    assert summarize(config, failed)['panels']['h']['paired_win_lift'] is None
    for defect in ('missing', 'wrong_phase', 'extra'):
        failed = copy.deepcopy(rows)
        records = failed[0]['candidate']['external']['0']['diagnostics']
        if defect == 'missing':
            records.pop()
        elif defect == 'wrong_phase':
            records[0]['phase'] = 'actions'
        else:
            records.append(copy.deepcopy(records[-1]))
        with pytest.raises(ValueError, match='opening coverage'):
            summarize(config, failed)
