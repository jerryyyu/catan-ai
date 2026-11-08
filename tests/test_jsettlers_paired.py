import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import jsettlers_paired as jp  # noqa: E402


def _game(name, won, board='b1'):
    return {'game': name, 'won': won, 'finished': True, 'server_board_sha': board, 'board_id': board,
            'server_first_player': 1, 'seat': 0, 'players': ['extrabot 1', 'robot 1', 'robot 2', 'robot 3'],
            'server_deck': ['knight'] * 3}


def _receipt(policy, wins, boards=None):
    boards = boards or ['b%d' % i for i in range(len(wins))]
    return {'jsettlers': {'seed': 7, 'patch_sha256': 'p', 'all_smart': True}, 'config': {'policy': policy, 'seed': 7},
            'games_requested': len(wins), 'bridge_errors': 0,
            'games': [_game('~botsOnly~%d' % i, w, b) for i, (w, b) in enumerate(zip(wins, boards))]}


def test_paired_difference_and_lcb():
    cand = _receipt('planner_pv2_tiles', [1, 1, 0, 1, 1, 0, 1, 1] * 20)
    ctrl = _receipt('planner_offer_accept', [0, 1, 0, 0, 1, 0, 0, 1] * 20)
    r = jp.pair(cand, ctrl, alpha=0.0125)
    assert r['compatible'] and r['games_paired'] == 160 and r['games_excluded'] == 0
    assert abs(r['mean_diff'] - 0.375) < 1e-9 and r['lcb'] > 0 and r['pass']


def test_setup_mismatch_is_excluded_and_too_many_exclusions_fail():
    cand = _receipt('planner_pv2_tiles', [1] * 40)
    ctrl = _receipt('planner_offer_accept', [0] * 40, boards=['x'] * 40)
    r = jp.pair(cand, ctrl, alpha=0.0125)
    assert r['games_paired'] == 0 and r['games_excluded'] == 40 and not r['pass']


def test_incompatible_receipts_and_same_policy_fail():
    cand = _receipt('planner_pv2_tiles', [1] * 10)
    ctrl = copy.deepcopy(cand)
    assert not jp.pair(cand, ctrl, alpha=0.0125)['pass']  # same policy
    ctrl['config']['policy'] = 'planner_offer_accept'
    ctrl['jsettlers']['seed'] = 8
    r = jp.pair(cand, ctrl, alpha=0.0125)
    assert not r['compatible'] and not r['pass']


def test_negative_or_zero_effect_fails():
    cand = _receipt('planner_pv2_tiles', [0, 1] * 50)
    ctrl = _receipt('planner_offer_accept', [0, 1] * 50)
    assert not jp.pair(cand, ctrl, alpha=0.0125)['pass']
