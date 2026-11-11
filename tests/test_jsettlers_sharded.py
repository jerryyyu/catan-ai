"""Sharded JSettlers tables: shard derivation, receipt merge, and exact pairing of merged receipts (no Java)."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import jsettlers_paired as jp  # noqa: E402
import jsettlers_sharded as js  # noqa: E402

SHA = 'c' * 64


def _config(policy='planner_pv2_tiles_va_t9', games=10, shards=3, port=8886, seed=20261120):
    return {'name': f'table-{policy}', 'policy': policy, 'seed': seed, 'games': games, 'shards': shards,
            'jsettlers': {'tag': 'release-2.6.10', 'commit': 'c514', 'seed': seed, 'all_smart': True,
                          'patch': {'file': 'runs/external/jsettlers-seed-v2.patch', 'sha256': 'p'}},
            'server': {'port': port, 'max_connections': 50, 'builtin_robots': 6, 'percent3p': 25,
                       'botgames_parallel': 1, 'botgames_wait_sec': 5, 'fast_pause_percent': 1,
                       'timeout_turn_sec': 120},
            'timeout_sec_per_game': 900, 'pass_bar': {'completed_games': games, 'bridge_errors': 0}}


def _board(seed, name):
    """The setup a seeded server would deal: a function of (shard seed, game name) only."""
    return f'{seed}/{name}'


def _game(name, seed, won):
    b = _board(seed, name)
    return {'game': name, 'won': bool(won), 'finished': True, 'winner': 0 if won else 1, 'seat': 0,
            'first_player': 1, 'server_first_player': 1, 'server_board_sha': b, 'board_id': b,
            'players': ['extrabot 1', 'robot 1', 'robot 2', 'robot 3'], 'server_deck': [b],
            'seed_logged_matches': True}


def _shard_receipt(cfg, wins, **over):
    """Shaped like jsettlers_smoke.main's receipt (only fields the merge and pairing read)."""
    seed = cfg['jsettlers']['seed']
    games = [_game(f'~botsOnly~{i}', seed, w) for i, w in enumerate(wins)]
    r = {'name': cfg['name'], 'config_sha256': SHA, 'config': cfg, 'games_requested': cfg['games'],
         'source_revision': 'abc', 'source_dirty': False,
         'build': {'manifest_sha256': 'm%d' % cfg['shard']['index'], 'jsettlers': {'tag': 'release-2.6.10'}},
         'bridge': {'policy': cfg['policy'], 'files_digest': 'f', 'checkpoint_model_hash': 'h',
                    'engine_backend': 'cython'},
         'jsettlers': {'patch_sha256': 'p', 'seed': seed, 'all_smart': True, 'build_dir': '/x'},
         'seeding': {'server_config': {'seed': str(seed)}, 'robot_strategies_logged': {}, 'seed_mismatches': []},
         'manifest_sha256': 'man', 'server_rc': 0, 'timed_out': False, 'elapsed_sec': 9.0 * len(wins),
         'bridge_retired': [], 'games_started': len(games), 'games_completed': len(games), 'truncated': 0,
         'unexpected_games': 0, 'our_wins': sum(map(bool, wins)), 'seat_counts': {'0': len(games)},
         'position_counts': {'3': len(games)}, 'decisions_by_type': {'build': 2 * len(games)},
         'fallbacks_total': 1, 'fallbacks_by_type': {'trade': 1},
         'fallbacks': [{'decision': 'trade', 'reason': 'x', 'game': '~botsOnly~0'}],
         'bridge_errors': 0, 'legal_mismatch_decisions': 0,
         'latency_ms': {'p50': 3.0, 'p95': 9.0, 'max': 20.0 + cfg['shard']['index'], 'n': 100},
         'think_ms': {'p50': 1.0, 'p95': 2.0, 'max': 3.0}, 'card_plays': {'knight': 1},
         'games': games, 'pass': True}
    r.update(over)
    return r


def _merged(config, wins_by_game, mutate=None):
    cfgs = js.shard_configs(config, SHA)
    receipts, i = [], 0
    for cfg in cfgs:
        receipts.append(_shard_receipt(cfg, wins_by_game[i:i + cfg['games']]))
        i += cfg['games']
    if mutate:
        mutate(receipts)
    return js.merge_receipts(config, SHA, receipts)


# ------------------------------------------------------------------ shard derivation

def test_shard_configs_seeds_games_ports():
    config = _config(games=5120, shards=3)
    cfgs = js.shard_configs(config, SHA)
    assert [c['games'] for c in cfgs] == [1707, 1707, 1706] and sum(c['games'] for c in cfgs) == 5120
    assert [c['jsettlers']['seed'] for c in cfgs] == [20261120000, 20261120001, 20261120002]
    assert [c['server']['port'] for c in cfgs] == [8886, 8986, 9086]
    assert all('shards' not in c and c['shard']['of'] == 3 and c['shard']['index'] == k for k, c in enumerate(cfgs))
    # everything else is the declared config, untouched
    keep = ('policy', 'seed', 'pass_bar', 'timeout_sec_per_game')
    assert all(c[k] == config[k] for c in cfgs for k in keep)
    assert config['shards'] == 3 and config['jsettlers']['seed'] == 20261120  # parent not mutated
    assert js.split_games(10, 4) == [3, 3, 2, 2] and js.split_games(8, 4) == [2] * 4


@pytest.mark.parametrize('change, match', [
    ({'shards': 1}, 'shards must be'), ({'shards': None}, 'shards must be'), ({'shards': True}, 'shards must be'),
    ({'shards': 1000}, 'seed map'), ({'shards': 11}, 'nonempty'),
])
def test_shard_configs_refuse_bad_k(change, match):
    config = dict(_config(games=10), **change)
    if change.get('shards') is None:
        config.pop('shards')
    with pytest.raises(js.ShardRefused, match=match):
        js.shard_configs(config, SHA)


def test_shard_configs_refuse_unseeded_parallel_ports():
    c = _config()
    c['jsettlers'].pop('seed')
    with pytest.raises(js.ShardRefused, match='seeded'):
        js.shard_configs(c, SHA)
    c = _config()
    c['server']['botgames_parallel'] = 2
    with pytest.raises(js.ShardRefused, match='botgames_parallel'):
        js.shard_configs(c, SHA)
    with pytest.raises(js.ShardRefused, match='65535'):
        js.shard_configs(_config(port=65500), SHA)
    with pytest.raises(js.ShardRefused, match='collides'):
        js.shard_configs(_config(port=8886), SHA, other_ports={8986: 'other.json'})
    with pytest.raises(js.ShardRefused, match='Java long'):
        js.shard_configs(_config(seed=2 ** 62), SHA)


def test_declared_ports_cover_real_configs_and_the_heldout_tables_shard_cleanly(tmp_path):
    ports = js.declared_ports(ROOT / 'experiments')
    assert 8886 in ports and 8885 in ports
    for name in ('jsettlers-va-heldout-t9.json', 'jsettlers-va-heldout-tiles.json'):
        path = ROOT / 'experiments' / name
        cfg = dict(json.loads(path.read_text()), shards=4)
        others = js.declared_ports(ROOT / 'experiments', exclude=path)
        cfgs = js.shard_configs(cfg, SHA, others)
        assert sum(c['games'] for c in cfgs) == cfg['games']
    # a sharded sibling config declares its shard ports too
    (tmp_path / 'a.json').write_text(json.dumps(_config(port=8001, shards=3)))
    assert set(js.declared_ports(tmp_path)) == {8001, 8101, 8201}


def test_cli_refuses_existing_out_before_running(tmp_path):
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(_config()))
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(js.ShardRefused, match='existing'):
        js.main(['--config', str(cfg), '--out', str(out)])
    assert list(out.iterdir()) == []


# ------------------------------------------------------------------ merge

def test_merge_prefixes_names_sums_totals_and_lists_shards():
    config = _config(games=10, shards=3)
    wins = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    m = _merged(config, wins)
    assert m['pass'] and m['merge_problems'] == []
    assert [g['game'] for g in m['games'][:5]] == ['s0:~botsOnly~0', 's0:~botsOnly~1', 's0:~botsOnly~2',
                                                  's0:~botsOnly~3', 's1:~botsOnly~0']
    assert m['games'][4]['shard'] == 1 and m['games'][4]['shard_game'] == '~botsOnly~0'
    assert m['games_requested'] == 10 and m['games_completed'] == 10 and m['our_wins'] == 5
    assert m['truncated'] == 0 and m['fallbacks_total'] == 3 and m['bridge_errors'] == 0
    assert m['seat_counts'] == {'0': 10} and m['card_plays'] == {'knight': 3}
    assert [f['game'] for f in m['fallbacks']] == ['s0:~botsOnly~0', 's1:~botsOnly~0', 's2:~botsOnly~0']
    assert m['latency_ms'] == {'max': 22.0, 'n': 300, 'p50': None, 'p95': None}
    assert m['jsettlers'] == {'patch_sha256': 'p', 'seed': 20261120, 'all_smart': True, 'shards': 3,
                              'shard_seed_function': 'base * 1000 + k', 'port_stride': 100,
                              'shard_seeds': [20261120000, 20261120001, 20261120002], 'shard_games': [4, 3, 3]}
    assert [(s['index'], s['seed'], s['port'], s['games_requested']) for s in m['shards']] == [
        (0, 20261120000, 8886, 4), (1, 20261120001, 8986, 3), (2, 20261120002, 9086, 3)]
    assert all(s['config_sha256'] == SHA and 'receipt_sha256' in s for s in m['shards'])
    assert m['win_rate'] == 0.5 and m['win_rate_wilson95'] is not None


@pytest.mark.parametrize('mutate', [
    lambda rs: rs[1].update(**{'pass': False}),
    lambda rs: rs[2].update(timed_out=True),
    lambda rs: rs[0].update(bridge_errors=1),
    lambda rs: rs[0].update(bridge_retired=[{'type': 'bridge_retired'}]),
    lambda rs: rs[1]['seeding'].update(seed_mismatches=['~botsOnly~2']),
    lambda rs: rs[1].update(source_revision='other'),
    lambda rs: rs[2]['bridge'].update(checkpoint_model_hash='other'),
    lambda rs: rs[1]['jsettlers'].update(seed=20261120),  # a shard run on the base seed
    lambda rs: rs[2].update(games_requested=2),
    lambda rs: rs[0].update(games_completed=3, truncated=1),
])
def test_merge_fails_closed(mutate):
    assert not _merged(_config(games=10, shards=3), [1] * 10, mutate)['pass']


def test_merge_refuses_missing_shard_receipt():
    config = _config(games=10, shards=3)
    cfgs = js.shard_configs(config, SHA)
    receipts = [_shard_receipt(c, [1] * c['games']) for c in cfgs[:2]]
    with pytest.raises(js.ShardRefused, match='2 shard receipts'):
        js.merge_receipts(config, SHA, receipts)
    with pytest.raises(js.ShardRefused):
        js.merge_receipts(config, SHA, receipts + [None])


# ------------------------------------------------------------------ pairing merged receipts

def _pair_inputs(k_cand=3, k_ctrl=3):
    wins_c = [1, 1, 0, 1, 1, 0, 1, 1] * 20
    wins_b = [0, 1, 0, 0, 1, 0, 0, 1] * 20
    cand = _merged(_config('planner_pv2_tiles_va_t9', games=160, shards=k_cand), wins_c)
    ctrl = _merged(_config('planner_pv2_tiles', games=160, shards=k_ctrl), wins_b)
    return cand, ctrl


def test_paired_accepts_two_merged_receipts_with_the_same_shard_design():
    cand, ctrl = _pair_inputs()
    r = jp.pair(cand, ctrl, alpha=0.05, expect_games=160)
    assert r['compatible'], r['problems']
    assert r['games_paired'] == 160 and r['games_excluded'] == 0
    assert abs(r['mean_diff'] - 0.375) < 1e-9 and r['pass']


def _unsharded(policy, wins, seed=20261120):
    return {'jsettlers': {'seed': seed, 'patch_sha256': 'p', 'all_smart': True},
            'config': {'policy': policy, 'seed': seed}, 'games_requested': len(wins),
            'games_completed': len(wins), 'truncated': 0, 'bridge_errors': 0,
            'games': [_game('~botsOnly~%d' % i, seed, w) for i, w in enumerate(wins)]}


def test_paired_refuses_sharded_vs_unsharded_explicitly():
    """Fails on main: main has no shard check (it only notices the game names differ)."""
    cand, _ = _pair_inputs()
    ctrl = _unsharded('planner_pv2_tiles', [0] * 160)
    for a, b in ((cand, ctrl), (ctrl, cand)):
        r = jp.pair(a, b, alpha=0.05, expect_games=160)
        assert not r['compatible'] and not r['pass']
        assert any('unsharded' in p for p in r['problems'])


def test_paired_refuses_mismatched_shard_designs():
    cand, ctrl = _pair_inputs(3, 4)
    r = jp.pair(cand, ctrl, alpha=0.05, expect_games=160)
    assert not r['pass'] and any(p.startswith('jsettlers.shards differs') for p in r['problems'])
    # Same K and identical names, but a different shard seed list: the names would pair, the luck would not.
    cand, ctrl = _pair_inputs()
    ctrl = copy.deepcopy(ctrl)
    ctrl['jsettlers']['shard_seeds'][1] += 7
    r = jp.pair(cand, ctrl, alpha=0.05, expect_games=160)
    assert not r['compatible'] and any('shard_seeds' in p for p in r['problems'])
    # A merge that recorded problems never confirms anything.
    cand, ctrl = _pair_inputs()
    cand['merge_problems'] = ['shards disagree on checkpoint']
    r = jp.pair(cand, ctrl, alpha=0.05, expect_games=160)
    assert not r['compatible'] and not r['pass']


def test_unsharded_pairing_is_unchanged():
    a = _unsharded('planner_pv2_tiles_va_t9', [1, 1, 0, 1] * 40)
    b = _unsharded('planner_pv2_tiles', [0, 1, 0, 0] * 40)
    assert jp.shard_problems(a, b) == []
    r = jp.pair(a, b, alpha=0.05, expect_games=160)
    assert r['compatible'] and r['pass']


def test_smoke_runner_refuses_a_sharded_config(tmp_path):
    import json as _json
    import jsettlers_smoke
    cfg = tmp_path / 'c.json'
    cfg.write_text(_json.dumps({'name': 'x', 'games': 4, 'shards': 2, 'jsettlers': {}}))
    with pytest.raises(SystemExit):
        jsettlers_smoke.main(['--config', str(cfg), '--out', str(tmp_path / 'o')])
    assert not (tmp_path / 'o').exists()
