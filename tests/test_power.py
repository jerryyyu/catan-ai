import copy
import math
from statistics import NormalDist

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


GATES = {'maximum_match_truncations': 0, 'minimum_paired_win_lift': .05}


def design(**overrides):
    config = {'panels': [{'name': 'a', 'seed': 100}, {'name': 'b', 'seed': 200}], 'seeds_per_panel': 512,
              'gates': dict(GATES), 'power': {'alternative_effect': .10, 'planning_sigma': .40}}
    config.update(overrides)
    return config


def test_admission_models_the_point_bar_not_a_test_against_zero():
    # An alternative equal to the bar clears it about half the time per panel,
    # whatever the cluster count; the zero-null formula would have admitted it.
    with pytest.raises(ValueError, match='underpowered'):
        check_design(design(power={'alternative_effect': .05, 'planning_sigma': .40}))
    record = check_design(design())
    assert record['pass_threshold'] == pytest.approx(.05)
    assert record['family_pass_probability'] >= .8
    assert record['family_pass_probability'] == pytest.approx(record['panel_pass_probability'] ** 2)
    needed = record['required_clusters']
    assert check_design(design(seeds_per_panel=needed))['family_pass_probability'] >= .8
    with pytest.raises(ValueError, match='underpowered'):
        check_design(design(seeds_per_panel=needed - 1))
    assert check_design({'panels': [], 'seeds_per_panel': 1}) is None


def test_lcb_gate_raises_the_threshold_under_each_inference():
    gates = {**GATES, 'minimum_paired_lcb': 0}
    hoeffding = check_design(design(gates=gates, power={'alternative_effect': .25, 'planning_sigma': .40}))
    assert hoeffding['pass_threshold'] == pytest.approx((2 * math.log(2 / .05) / 512) ** .5)
    normal = check_design(design(gates=gates, inference='paired_normal'))
    assert normal['pass_threshold'] == pytest.approx(max(.05, NormalDist().inv_cdf(1 - .025) * .40 / 512 ** .5))
    strict = check_design(design(gates={**gates, 'minimum_paired_lcb': .03}, inference='paired_normal'))
    assert strict['pass_threshold'] == pytest.approx(.03 + NormalDist().inv_cdf(1 - .025) * .40 / 512 ** .5)
    # The Hoeffding margin at 512 clusters (~.12) makes a +10 pp alternative unreachable.
    with pytest.raises(ValueError, match='underpowered'):
        check_design(design(gates=gates))


@pytest.mark.parametrize('block', [
    {'alternative_effect': math.inf, 'planning_sigma': .4},
    {'alternative_effect': math.nan, 'planning_sigma': .4},
    {'alternative_effect': .1, 'planning_sigma': math.inf},
    {'alternative_effect': .1, 'planning_sigma': 0},
    {'alternative_effect': .1, 'planning_sigma': 1.5},
    {'alternative_effect': 2, 'planning_sigma': .4},
    {'alternative_effect': .1, 'planning_sigma': .4, 'power': 1},
    {'alternative_effect': '.1', 'planning_sigma': .4},
    {'planning_sigma': .4},
])
def test_admission_rejects_nonfinite_or_out_of_range_inputs(block):
    with pytest.raises(ValueError):
        check_design(design(power=block, seeds_per_panel=2))


def test_admission_rejects_unknown_inference_bad_counts_and_legacy_blocks():
    with pytest.raises(ValueError, match='inference'):
        check_design(design(inference='bootstrap'))
    with pytest.raises(ValueError, match='inference'):
        check_design(design(inference='bootstrap', gates={**GATES, 'minimum_paired_lcb': 0}))
    for count in (1, 2.5, True):
        with pytest.raises(ValueError, match='seeds_per_panel'):
            check_design(design(seeds_per_panel=count))
    with pytest.raises(ValueError, match='legacy'):
        check_design(design(power={'target_effect': .05, 'planning_sigma': .40}))


def test_zero_null_helpers_reject_nonfinite_inputs():
    with pytest.raises(ValueError):
        required_clusters(.4, math.inf)
    with pytest.raises(ValueError):
        minimum_detectable_effect(math.nan, 10)


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
    assert panel['zero_null_mde_80'] > .5
    assert 'minimum_detectable_effect_80' not in panel
    assert before['pass'] is True


def test_parallel_shards_drop_the_power_block_and_the_parent_checks_it():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('parallel_search_eval', Path('scripts/parallel_search_eval.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = design(stage='dev', seeds_per_panel=600)
    shard = module.shard_config(config, 3)
    assert 'power' not in shard and shard['seeds_per_panel'] == 1 and shard['panels'][0]['seed'] == 103
    assert check_design(config)['family_pass_probability'] >= .8
    with pytest.raises(ValueError, match='underpowered'):
        check_design({**config, 'seeds_per_panel': 8})
