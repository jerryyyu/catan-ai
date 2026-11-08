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


def test_every_screen_and_external_row_is_drawn():
    page = build.render(REG)
    marks = sum(len(s['tables']) for s in REG['screens'] if s['vs'] == REG['production']['served'])
    assert page.count('<circle') == marks + len(REG['external'])
    for s in REG['screens']:
        assert s['id'] in page
