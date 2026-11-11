"""Finalizing a run killed externally after play, and the analyzer's predeclared --allow-interrupted (2026-11-11)."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import jsettlers_paired as jp  # noqa: E402
import jsettlers_smoke as smoke  # noqa: E402
from test_jsettlers_paired import _receipt  # noqa: E402


def _finalized(r, interrupted='~botsOnly~0', last=None, extra_unfinished=()):
    r = copy.deepcopy(r)
    for g in r['games']:
        if g['game'] == interrupted or g['game'] in extra_unfinished:
            g['finished'], g['won'] = False, None
    unfinished = [g['game'] for g in r['games'] if not g['finished']]
    r['games_completed'] = len(r['games']) - len(unfinished)
    r['truncated'] = len(unfinished)
    r['server_rc'] = None
    last = last or interrupted
    r['finalized'] = {'reason': 'killed', 'unfinished_games': unfinished, 'last_started_game': last,
                      'interrupted_game': last if unfinished == [last] else None}
    return r


def _pair_inputs():
    cand = _receipt('planner_pv2_tiles_va_t9', [1, 1, 0, 1, 1, 0, 1, 1] * 20)
    ctrl = _receipt('planner_pv2_tiles', [0, 1, 0, 0, 1, 0, 0, 1] * 20)
    return cand, ctrl


def test_interrupted_last_game_is_excluded_only_when_declared():
    cand, ctrl = _pair_inputs()
    fc, fb = _finalized(cand), _finalized(ctrl)
    strict = jp.pair(fc, fb, alpha=.05, expect_games=160)
    assert not strict['compatible'] and not strict['pass']           # unchanged default
    r = jp.pair(fc, fb, alpha=.05, expect_games=160, allow_interrupted=True)
    assert r['compatible'], r['problems']
    assert r['games_excluded'] == 1 and r['games_paired'] == 159
    assert r['interrupted_excluded'] == {'candidate': '~botsOnly~0', 'control': '~botsOnly~0'}


def test_one_side_interrupted_excludes_that_pair():
    cand, ctrl = _pair_inputs()
    r = jp.pair(_finalized(cand), ctrl, alpha=.05, expect_games=160, allow_interrupted=True)
    assert r['compatible'] and r['games_excluded'] == 1


@pytest.mark.parametrize('kw', [
    {'extra_unfinished': ('~botsOnly~5',)},          # a second unfinished game is never excused
    {'last': '~botsOnly~9'},                          # unfinished game is not the last one started
])
def test_anything_beyond_the_single_last_started_game_stays_incompatible(kw):
    cand, ctrl = _pair_inputs()
    r = jp.pair(_finalized(cand, **kw), ctrl, alpha=.05, expect_games=160, allow_interrupted=True)
    assert not r['compatible'] and not r['pass']


def test_unfinalized_receipts_are_unaffected_by_the_flag():
    cand, ctrl = _pair_inputs()
    cand['games'][3]['finished'] = False
    cand['games_completed'] -= 1
    cand['truncated'] = 1
    assert not jp.pair(cand, ctrl, alpha=.05, expect_games=160, allow_interrupted=True)['compatible']


def test_finalize_seals_once_from_the_run_and_records_the_interruption(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'build-manifest.json').write_text('{}')
    sha = smoke.jsettlers_build.sha256(out / 'build-manifest.json')
    (out / 'manifest.json').write_text(json.dumps({'config': {'seed': 1, 'jsettlers': {}}, 'games_requested': 3,
                                                   'build': {'manifest_sha256': sha}, 'source_revision': 'abc'}))
    for name in ('decisions.jsonl', 'bridge.jsonl'):
        (out / name).write_text('')
    (out / 'server.log').write_text('catanai-start game=~botsOnly~3 first=0\ncatanai-start game=~botsOnly~2 first=1\n'
                                    'catanai-start game=~botsOnly~1 first=2\ncatanai-dice game=~botsOnly~1 roll=1\n')
    games = [{'game': '~botsOnly~3', 'finished': True, 'won': True}, {'game': '~botsOnly~2', 'finished': True, 'won': False},
             {'game': '~botsOnly~1', 'finished': False, 'won': None}]
    monkeypatch.setattr(smoke, 'jsettlers_settings', lambda c: {})
    monkeypatch.setattr(smoke, 'build_receipt', lambda *a: {'games': games, 'games_completed': 2, 'pass': False,
                                                             'server_rc': a[5], 'elapsed_sec': a[7]})
    r = smoke.finalize(out, 'reaper killed the runner after play')
    assert r['finalized']['interrupted_game'] == '~botsOnly~1' and r['server_rc'] is None and r['elapsed_sec'] is None
    assert r['finalized']['manifest_source_revision'] == 'abc' and json.loads((out / 'receipt.json').read_text()) == r
    with pytest.raises(smoke.RunRefused, match='already has a receipt'):
        smoke.finalize(out, 'again')
    (out / 'receipt.json').unlink()
    (out / 'build-manifest.json').write_text('{"changed": 1}')
    with pytest.raises(smoke.RunRefused, match='build manifest changed'):
        smoke.finalize(out, 'x')
