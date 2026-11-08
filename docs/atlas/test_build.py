import copy
import json
from pathlib import Path

import build

REG = json.loads((Path(__file__).parent / 'registry.json').read_text())


def test_registry_is_consistent():
    assert build.check_registry(REG) == []


def test_interval_must_contain_point():
    reg = copy.deepcopy(REG)
    reg['screens'][1]['tables']['tactical'] = [0.5, 0.0, 0.1]
    assert any('outside' in p for p in build.check_registry(reg))


def test_verdict_and_duplicate_ids_are_checked():
    reg = copy.deepcopy(REG)
    reg['screens'][0]['verdict'] = 'maybe'
    reg['screens'].append(copy.deepcopy(reg['screens'][1]))
    problems = build.check_registry(reg)
    assert any('verdict' in p for p in problems) and any('duplicate' in p for p in problems)


def test_every_policy_row_is_in_the_matrix():
    page = build.render(REG)
    for pol in REG['policies']:
        assert pol['name'] in page
    assert 'Every policy at a glance' in page
    assert page.index('Every policy at a glance') < page.index('Evaluation strategy')


def test_policy_intervals_are_checked():
    reg = copy.deepcopy(REG)
    reg['policies'][1]['vs_prod']['tactical'] = [0.5, 0.0, 0.1]
    reg['policies'][0]['external']['jsettlers'] = [0.9, 0.1, 0.2, 512]
    problems = build.check_registry(reg)
    assert any('bad interval' in p for p in problems) and any('bad win share' in p for p in problems)
