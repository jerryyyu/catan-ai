import copy

import pytest

from catan_ai.power import (check_design, cluster_sigma, minimum_detectable_effect, panel_differences,
                            required_clusters)
from catan_ai.search_eval import summarize


def test_mde_and_required_clusters_invert_each_other():
    sigma, alpha = .35, .025
    needed = required_clusters(sigma, .05, alpha)
    assert minimum_detectable_effect(sigma, needed, alpha) <= .05
    assert minimum_detectable_effect(sigma, needed - 1, alpha) > .05


def test_eight_cluster_screens_cannot_see_a_five_point_lift():
    # The historical DEV design: 8 or 16 boards per panel, two panels.
    for clusters in (8, 16):
        assert minimum_detectable_effect(.25, clusters, .025) > .10


def test_check_design_refuses_underpowered_configs_and_accepts_powered_ones():
    config = {'panels': [{'name': 'a'}, {'name': 'b'}], 'seeds_per_panel': 8,
              'power': {'target_effect': .05, 'planning_sigma': .36}}
    with pytest.raises(ValueError, match='underpowered'):
        check_design(config)
    config['seeds_per_panel'] = check_design({**config, 'seeds_per_panel': 10 ** 6})
    assert check_design(config) == config['seeds_per_panel']
    assert check_design({'panels': [], 'seeds_per_panel': 1}) is None


def test_panel_differences_average_seats_within_a_board():
    rows = [{'panel': 'p', 'seed': 1, 'seat': s, 'candidate': {'winner': s if s < 2 else 9}, 'baseline': {'winner': 9}}
            for s in range(4)]
    assert panel_differences(rows, 'p') == [.5]
    assert panel_differences(rows, 'other') == []


def test_summary_reports_descriptive_power_without_changing_the_verdict():
    config = {'panels': [{'name': 'h', 'seed': 10, 'opponents': ['heuristic'] * 3}], 'seeds_per_panel': 2,
              'requires_search': False,
              'gates': {'maximum_match_truncations': 0, 'minimum_paired_win_lift': .05,
                        'minimum_search_coverage': .8, 'maximum_p95_decision_seconds': 2,
                        'maximum_rollout_truncation_rate': .01}}
    rows = []
    for seed in (10, 11):
        for seat in range(4):
            rows.append({'panel': 'h', 'seed': seed, 'seat': seat,
                         'candidate': {'winner': seat if seed == 10 else (seat + 1) % 4, 'truncated': False, 'search': {},
                                       'decision_seconds': [[.01]] * 4},
                         'baseline': {'winner': (seat + 1) % 4, 'truncated': False}})
    before = summarize(copy.deepcopy(config), copy.deepcopy(rows))
    panel = before['panels']['h']
    assert panel['seed_cluster_differences'] == [1.0, 0.0]
    assert panel['cluster_sd'] == pytest.approx(cluster_sigma([1.0, 0.0]))
    assert panel['minimum_detectable_effect_80'] > .5
    assert before['pass'] is True
