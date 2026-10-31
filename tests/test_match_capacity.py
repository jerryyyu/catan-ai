from pathlib import Path
import runpy

import pytest


validity = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                             'scripts/benchmark_match_capacity.py'))['validity']


def test_capacity_cases_cover_trained_candidate_and_external_in_all_seats():
    cases = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              'scripts/benchmark_match_capacity.py'))['cases']
    selected = cases('planner_parameter', 1479900)
    assert len(selected) == 8
    for seat in range(4):
        for offset, name in enumerate(('planner_parameter', 'external')):
            case = selected[seat * 2 + offset]
            assert case['seed'] == 1479900 + seat % 2
            assert case['policies'][seat] == name
            assert case['policies'][:seat] + case['policies'][seat + 1:] == ['builder', 'external', 'expander']
    assert cases()[0]['policies'][0] == 'planner_robber'


def game():
    return {'winner': 0, 'truncated': False, 'step_seconds': [[.2] for _ in range(4)],
            'turn_timings': [{'complete': True, 'seconds': .8}],
            'external': {'0': {'identity': {'test': True}, 'diagnostics': [
                {'fallback': False, 'budget_failure': False, 'phase': 'main',
                 'turn': 1, 'seconds': .2}]}}}


@pytest.mark.parametrize('defect', ['none', 'missing_seat', 'fault', 'step',
                                    'turn', 'compute', 'nan', 'truncated',
                                    'missing_decision', 'missing_identity',
                                    'invalid_winner', 'negative_diagnostic'])
def test_capacity_requires_complete_fault_free_timed_games(defect):
    result = game()
    if defect == 'missing_seat':
        result['external'] = {}
    elif defect == 'fault':
        result['external']['0']['diagnostics'][0]['fallback'] = True
    elif defect == 'step':
        result['step_seconds'][0] = [2.001]
    elif defect == 'turn':
        result['turn_timings'][0]['seconds'] = 10.001
    elif defect == 'compute':
        result['external']['0']['diagnostics'][0]['seconds'] = 10.001
    elif defect == 'nan':
        result['step_seconds'][0] = [float('nan')]
    elif defect == 'truncated':
        result['truncated'] = True
    elif defect == 'missing_decision':
        result['step_seconds'][0].append(.1)
    elif defect == 'missing_identity':
        del result['external']['0']['identity']
    elif defect == 'invalid_winner':
        result['winner'] = None
    elif defect == 'negative_diagnostic':
        result['external']['0']['diagnostics'][0]['seconds'] = -.1
    assert validity(result, ['external', 'tactical', 'builder', 'expander'])['pass'] == (
        defect == 'none')
