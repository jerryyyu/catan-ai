import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import stac_paired as sp  # noqa: E402

SETUP = {'board_sha': 'b', 'deck': '[1]', 'first_player': 0, 'seats': ['x', 'y', 'CatanAi', 'z'], 'seat': 2}


def _receipt(policy, pomcp_wins, neg_wins=()):
    games = [{'game': f'P_{i}', 'opponent': 'pomcp_ts_cr', 'finished': True, 'won': w, **SETUP}
             for i, w in enumerate(pomcp_wins)]
    games += [{'game': f'N_{i}', 'opponent': 'stac_negotiator', 'finished': True, 'won': w, **SETUP}
              for i, w in enumerate(neg_wins)]
    return {'games_requested': len(games), 'config': {'policy': policy, 'blocks': [1], 'agents': {}, 'seed': 1},
            'stac': {'seed': 5, 'patch_sha256': 'p', 'player_trading': 'chat'}, 'runs': [{'timed_out': False}],
            'total': {'games_completed': len(games), 'truncated': 0, 'bridge_errors': 0, 'chat_trades': {},
                      'observability_violations': [], 'seed_mismatches': [], 'stac_results_disagree': []},
            'games': games}


def _expect(r):
    out = {}
    for g in r['games']:
        out[g['opponent']] = out.get(g['opponent'], 0) + 1
    return out


def test_pairs_shares_and_difference():
    a = _receipt('planner_pv2_tiles', [True, True, False, False], [False, True])
    b = _receipt('planner_offer_accept', [True, False, False, False], [False, False])
    r = sp.compare(a, b, _expect(a))
    t = r['tables']['pomcp_ts_cr']
    assert r['healthy'] and t['a']['wins'] == 2 and t['b']['wins'] == 1 and t['paired_a_minus_b']['mean_diff'] == 0.25
    assert not t['a']['retired'] and r['tables']['stac_negotiator']['a']['wins'] == 1


def test_coverage_is_declared_not_inferred():
    # Codex HOLD counterexample: 2 records each but claimed 768 requested/completed.
    a = _receipt('planner_pv2_tiles', [True, False])
    b = _receipt('planner_offer_accept', [True, False])
    for r in (a, b):
        r['games_requested'] = r['total']['games_completed'] = 768
    r = sp.compare(a, b, {'pomcp_ts_cr': 256, 'stac_negotiator': 512})
    assert not r['healthy'] and any('per opponent' in p for p in r['problems'])
    # two matching smaller receipts are not the declared design either
    small_a, small_b = _receipt('planner_pv2_tiles', [True] * 4), _receipt('planner_offer_accept', [False] * 4)
    assert not sp.compare(small_a, small_b, {'pomcp_ts_cr': 256, 'stac_negotiator': 512})['healthy']


def test_underfilled_opponent_block_fails():
    a = _receipt('planner_pv2_tiles', [True] * 4, [False] * 2)
    b = _receipt('planner_offer_accept', [True] * 4, [False] * 2)
    r = sp.compare(a, b, {'pomcp_ts_cr': 4, 'stac_negotiator': 3})
    assert not r['healthy']


@pytest.mark.parametrize('key', ['observability_violations', 'seed_mismatches', 'stac_results_disagree',
                                 'truncated', 'bridge_errors', 'games_completed', 'chat_trades'])
def test_missing_health_evidence_fails(key):
    a = _receipt('planner_pv2_tiles', [True, False])
    b = _receipt('planner_offer_accept', [True, False])
    del b['total'][key]
    r = sp.compare(a, b, _expect(a))
    assert not r['healthy'] and any(key in p for p in r['problems'])


def test_duplicate_names_and_unfinished_games_fail():
    a = _receipt('planner_pv2_tiles', [True, False, True])
    b = _receipt('planner_offer_accept', [True, False, True])
    dup = copy.deepcopy(b)
    dup['games'][2]['game'] = 'P_1'
    assert any('duplicate' in p for p in sp.compare(a, dup, _expect(a))['problems'])
    unfinished = copy.deepcopy(b)
    unfinished['games'][0]['finished'] = False
    assert not sp.compare(a, unfinished, _expect(a))['healthy']
    truncated = copy.deepcopy(b)
    truncated['total']['truncated'] = 1
    assert not sp.compare(a, truncated, _expect(a))['healthy']


def test_compat_policy_and_timeouts():
    a = _receipt('planner_pv2_tiles', [True, False])
    assert not sp.compare(a, copy.deepcopy(a), _expect(a))['healthy']  # same policy
    b = _receipt('planner_offer_accept', [True, False])
    bad = copy.deepcopy(b)
    bad['total']['chat_trades'] = {'offer_timeout': 1}
    assert any('offer timeouts' in p for p in sp.compare(a, bad, _expect(a))['problems'])
    other = copy.deepcopy(b)
    other['stac']['seed'] = 6
    assert any('stac.seed differs' in p for p in sp.compare(a, other, _expect(a))['problems'])
    with pytest.raises(ValueError):
        sp.compare(a, b, {})


def test_setup_mismatch_is_excluded_and_counted():
    a = _receipt('planner_pv2_tiles', [True] * 10)
    b = _receipt('planner_offer_accept', [False] * 10)
    b['games'][0]['seat'] = 1
    r = sp.compare(a, b, _expect(a))
    assert r['games_excluded'] == 1 and not r['healthy']  # 1 of 10 > 5%
    b2 = _receipt('planner_offer_accept', [False] * 10)
    assert sp.compare(a, b2, _expect(a))['healthy']


def _blocked(policy, pomcp_wins, neg_wins, neg_errors=0):
    """A receipt with real block declarations and per-block evidence (as stac_match writes)."""
    blocks = [{'name': 'LadPomcpA', 'opponent': 'pomcp_ts_cr', 'games': len(pomcp_wins)},
              {'name': 'LadNegA', 'opponent': 'stac_negotiator', 'games': len(neg_wins)}]
    games = [{'game': f'P_{i}', 'block': 'LadPomcpA', 'opponent': 'pomcp_ts_cr', 'finished': True, 'won': w, **SETUP}
             for i, w in enumerate(pomcp_wins)]
    games += [{'game': f'N_{i}', 'block': 'LadNegA', 'opponent': 'stac_negotiator', 'finished': True, 'won': w, **SETUP}
              for i, w in enumerate(neg_wins)]

    def ev(n, errors=0):
        return {'games': n, 'games_completed': n, 'truncated': 0, 'bridge_errors': errors,
                'chat_trades': {'offer_timeout': 0}, 'observability_violations': [], 'seed_mismatches': [],
                'stac_results_disagree': []}
    per = {'LadPomcpA': ev(len(pomcp_wins)), 'LadNegA': ev(len(neg_wins), neg_errors)}
    total = ev(len(games), neg_errors)
    return {'games_requested': len(games), 'config': {'policy': policy, 'blocks': blocks, 'agents': {}, 'seed': 1},
            'stac': {'seed': 5, 'patch_sha256': 'p', 'player_trading': 'chat'},
            'runs': [{'block': 'LadPomcpA', 'timed_out': False}, {'block': 'LadNegA', 'timed_out': False}],
            'blocks': per, 'total': total, 'games': games}


def test_pomcp_only_candidate_pairs_with_a_full_ladder_receipt():
    """Owner 2026-11-11: negotiator dropped from the promotion bar; pair on the declared POMCP blocks only."""
    full = _blocked('planner_pv2_tiles', [True, False, False, True], [False, False, True], neg_errors=2)
    pomcp = _blocked('planner_pv2_tiles_va_t9', [True, True, False, True], [])
    pomcp['config']['blocks'] = pomcp['config']['blocks'][:1]
    pomcp['blocks'] = {'LadPomcpA': pomcp['blocks']['LadPomcpA']}
    pomcp['runs'] = pomcp['runs'][:1]
    r = sp.compare(pomcp, full, {'pomcp_ts_cr': 4})
    assert r['healthy'] and set(r['tables']) == {'pomcp_ts_cr'} and r['restricted_from']['b'] == ['LadNegA']
    assert r['tables']['pomcp_ts_cr']['paired_a_minus_b']['mean_diff'] == 0.25   # neg bridge errors no longer count
    # the full design is unchanged: both opponents declared, the negotiator errors still fail health
    full_b = _blocked('planner_offer_accept', [True] * 4, [False] * 3)
    assert not sp.compare(full, full_b, {'pomcp_ts_cr': 4, 'stac_negotiator': 3})['healthy']


def test_restriction_keeps_pomcp_health_and_compat():
    full = _blocked('planner_pv2_tiles', [True, False], [False])
    bad = _blocked('planner_pv2_tiles_va_t9', [True, False], [])
    bad['blocks']['LadPomcpA']['bridge_errors'] = 1
    assert not sp.compare(bad, full, {'pomcp_ts_cr': 2})['healthy']
    other = _blocked('planner_pv2_tiles_va_t9', [True, False], [])
    other['config']['blocks'][0]['games'] = 3
    assert not sp.compare(other, full, {'pomcp_ts_cr': 2})['healthy']   # POMCP block declarations must match
    missing = _blocked('planner_pv2_tiles_va_t9', [True, False], [False])
    del missing['blocks']['LadPomcpA']
    assert not sp.compare(missing, full, {'pomcp_ts_cr': 2})['healthy']  # no per-block evidence, no pass
