import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import stac_paired as sp  # noqa: E402

SETUP = {'board_sha': 'b', 'deck': '[1]', 'first_player': 0, 'seats': ['x', 'y', 'CatanAi', 'z'], 'seat': 2}


def _receipt(policy, wins):
    games = [{'game': f'P_{i}', 'opponent': 'pomcp_ts_cr', 'finished': True, 'won': w, **SETUP} for i, w in enumerate(wins)]
    return {'games_requested': len(games), 'config': {'policy': policy, 'blocks': [1], 'agents': {}, 'seed': 1},
            'stac': {'seed': 5, 'patch_sha256': 'p', 'player_trading': 'chat'}, 'runs': [{'timed_out': False}],
            'total': {'games_completed': len(games), 'bridge_errors': 0, 'chat_trades': {}, 'observability_violations': [],
                      'seed_mismatches': [], 'stac_results_disagree': []}, 'games': games}


def test_pairs_shares_and_difference():
    a = _receipt('planner_pv2_tiles', [True, True, False, False])
    b = _receipt('planner_offer_accept', [True, False, False, False])
    r = sp.compare(a, b)
    t = r['tables']['pomcp_ts_cr']
    assert r['healthy'] and t['a']['wins'] == 2 and t['b']['wins'] == 1 and t['paired_a_minus_b']['mean_diff'] == 0.25
    assert not t['a']['retired']


def test_fails_closed_on_health_and_compat():
    a = _receipt('planner_pv2_tiles', [True, False])
    b = _receipt('planner_pv2_tiles', [True, False])
    assert not sp.compare(a, b)['healthy']  # same policy
    b = _receipt('planner_offer_accept', [True, False])
    bad = copy.deepcopy(b)
    bad['total']['chat_trades'] = {'offer_timeout': 1}
    assert any('offer timeouts' in p for p in sp.compare(a, bad)['problems'])
    other = copy.deepcopy(b)
    other['stac']['seed'] = 6
    assert any('stac.seed differs' in p for p in sp.compare(a, other)['problems'])


def test_setup_mismatch_is_excluded_and_counted():
    a = _receipt('planner_pv2_tiles', [True] * 10)
    b = _receipt('planner_offer_accept', [False] * 10)
    b['games'][0]['seat'] = 1
    r = sp.compare(a, b)
    assert r['games_excluded'] == 1 and not r['healthy']  # 1 of 10 > 5%
