"""StacSettlers match runner and build: config refusals, simulation config, log parsing, compare."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools' / 'jsettlers'))  # same import root as the other bridge tests (CI)

import stac_build as sb  # noqa: E402
import stac_match as sm  # noqa: E402

SMOKE = json.loads((ROOT / 'experiments' / 'stac-smoke.json').read_text())
DETERMINISM = json.loads((ROOT / 'experiments' / 'stac-determinism.json').read_text())


def test_configs_pin_the_build_and_pass_the_checks():
    for config in (SMOKE, DETERMINISM):
        stac = sm.check_config(config)
        assert stac['commit'] == sb.STAC_COMMIT and stac['repo'] == sb.STAC_REPO
        assert len(stac['patch']['sha256']) == 64
        assert stac['seed'] is not None
    assert DETERMINISM['stac']['patch'] == SMOKE['stac']['patch']


def test_smoke_declares_the_paper_setting_and_a_cap():
    mcts = SMOKE['agents']['pomcp_ts_cr']
    assert {'MCTS_ITERATIONS:10000', 'MCTS_THREADS:4', 'MCTS_POMCP', 'MCTS_PUCT', 'MCTS_Cp:4.0',
            'MCTS_HUMAN_ACTION_TYPE_PDF_CONDITIONED_SEEDING',
            'MCTS_HUMAN_ACTION_TYPE_CONDITIONED_DIST_ROLLOUTS', 'NO_TRADES'} <= set(mcts['flags'])
    assert mcts['time_cap_ms'] == 10000
    assert {b['opponent']: b['games'] for b in SMOKE['blocks']} == {'pomcp_ts_cr': 8, 'jsettlers': 8}


@pytest.mark.parametrize('mutate, message', [
    (lambda c: c['stac'].update(fully_observable=True), 'fully_observable'),
    (lambda c: c['stac'].update(observable_vp=True), 'observable_vp'),
    (lambda c: c['stac'].update(chat_negotiation=True), 'chat_negotiation'),
    (lambda c: c['stac'].update(no_player_trades=False), 'no_player_trades'),
    (lambda c: c['stac'].update(commit='0' * 40), 'pins StacSettlers'),
    (lambda c: c['stac'].pop('patch'), 'patch'),
    (lambda c: c['stac'].update(seed='7'), 'integer'),
    (lambda c: c['agents']['pomcp_ts_cr']['flags'].remove('NO_TRADES'), 'NO_TRADES'),
    (lambda c: c['agents']['pomcp_ts_cr'].pop('time_cap_ms'), 'time_cap_ms'),
    (lambda c: c['agents']['pomcp_ts_cr']['flags'].append('MCTS_TIME_LIMIT_MS:5'), 'time_cap_ms'),
    (lambda c: c['blocks'].append(dict(c['blocks'][0])), 'duplicate'),
    (lambda c: c['blocks'][0].update(opponent='stac'), 'unknown opponent'),
    (lambda c: c['blocks'][0].update(name='bad name'), 'block name'),
])
def test_config_refusals(mutate, message):
    config = copy.deepcopy(SMOKE)
    mutate(config)
    with pytest.raises(sm.RunRefused, match=message):
        sm.check_config(config)


def test_simulation_config_text():
    block = {'name': 'StacSmokePomcp', 'opponent': 'pomcp_ts_cr', 'games': 8}
    text = sm.simulation_config(block, SMOKE['agents'])
    lines = text.splitlines()
    assert lines[:7] == ['Games=8', 'Log=false', 'ChatNegotiation=false', 'FullyObservable=false',
                         'ObservableVP=false', '~', 'StacSmokePomcp']
    assert 'Trades=false' in lines
    assert 'Agent=1,CatanAi,class:catanai.stac.CatanAiStacFactory' in lines
    agent = next(line for line in lines if line.startswith('Agent=3,POMCPTSCR,mcts,'))
    assert agent.endswith('|NO_TRADES|MCTS_TIME_LIMIT_MS:10000')
    jset = sm.simulation_config({'name': 'J', 'opponent': 'jsettlers', 'games': 1}, SMOKE['agents'])
    assert 'Agent=3,JSettlers,jsettlers' in jset.splitlines()


def test_jvm_props_carry_seed_trades_and_turn_limit():
    props = sm.jvm_props(SMOKE['stac'])
    assert '-Dstac.catanai.noPlayerTrades=Y' in props
    assert f'-Dstac.catanai.seed={SMOKE["stac"]["seed"]}' in props
    assert '-Dstac.catanai.forceEndTurnSec=30' in props
    assert not any('seed' in p for p in sm.jvm_props({**SMOKE['stac'], 'seed': None}))


def test_seed_derivation_matches_the_java_patch():
    # Logged by the patched STAC server: catanai-seed game=IntOne_0 game_seed=6058317579988585315 (base 20261108).
    assert sm.derive_seed(20261108, 'IntOne_0', 'game') == 6058317579988585315


LOG = """noise
catanai-config seed=20261108 noPlayerTrades=true forceEndTurnSec=30
catanai-seed game=B_0 game_seed=42
catanai-start game=B_0 first=2 seats=JSettlers_1|JSettlers_3|JSettlers_2|CatanAi hexes=[9, 6] numbers=[-1, 11] ports=null deck=[1, 0]
catanai-observability game=B_0 fully_observable=false observable_vp=false chat_negotiations=false no_player_trades=true
catanai-dice game=B_0 roll=0 turn=1 pn=2 dice=6,2
catanai-dice game=B_0 roll=1 turn=2 pn=3 dice=1,4
catanai-mcts game=B_0 pn=1 ms=10120 iterations=4210 max=10000 cap_ms=10000 cap_hit=true
catanai-mcts game=B_0 pn=1 ms=900 iterations=10000 max=10000 cap_ms=10000 cap_hit=false
catanai-force-end game=B_0 pn=1 state=20 turn=7 name=POMCPTSCR_1
catanai-trade-blocked game=B_0 offering=1 accepting=3
catanai-end game=B_0 winner=3 first=2 turns=95 rounds=24 vp=7,6,9,10 seats=JSettlers_1|JSettlers_3|JSettlers_2|CatanAi
"""


def test_server_facts_parse_every_catanai_line(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text(LOG)
    facts = sm.server_facts(log)
    g = facts['B_0']
    assert facts['_config']['seed'] == '20261108'
    assert g['game_seed_logged'] == 42 and g['first_player'] == 2
    assert g['seats'] == ['JSettlers_1', 'JSettlers_3', 'JSettlers_2', 'CatanAi']
    assert g['hexes'] == '[9, 6]' and g['ports'] == 'null' and g['deck'] == '[1, 0]'
    assert g['dice'] == ['0:1:2:6,2', '1:2:3:1,4']
    assert [m['cap_hit'] for m in g['mcts']] == [True, False]
    assert g['force_ends'] == [{'pn': 1, 'state': 20, 'turn': 7}]
    assert g['trade_blocked'] == 1
    assert g['end']['winner'] == 3 and g['end']['vp'] == [7, 6, 9, 10]
    assert g['observability']['fully_observable'] == 'false'


def _block(tmp_path, log_text, decisions):
    d = tmp_path / 'B'
    (d / 'results' / 'r').mkdir(parents=True)
    (d / 'run.log').write_text(log_text)
    (d / 'decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decisions))
    header = 'Game name\tDate/time\t' + '\t'.join(f'Name{i}\tWinner{i}\tVP{i}\tOffers{i}\tSuccessfulOffers{i}\tTrades{i}'
                                                    for i in range(4))
    row = 'B_0\tx\tCatanAi\t1\t10\t0\t0\t0\tJSettlers_3\t0\t6\t0\t0\t0\tJSettlers_1\t0\t7\t0\t0\t0\tJSettlers_2\t0\t9\t0\t0\t0'
    (d / 'results' / 'r' / 'results.txt').write_text(header + '\n' + row + '\n')
    return d


DECISIONS = [{'type': 'game_start', 'game': 'B_0', 'seat': 3},
             {'type': 'decision', 'game': 'B_0', 'decision': 'roll', 'cmd': 'roll', 'latency_ms': 2.0,
              'think_ms': 1.0, 'turn_count': 2},
             {'type': 'decision', 'game': 'B_0', 'decision': 'actions', 'cmd': 'error', 'latency_ms': 3.0,
              'error': 'boom', 'turn_count': 2},
             {'type': 'fallback', 'game': 'B_0', 'decision': 'actions', 'cmd': 'error', 'reason': 'bridge error: boom',
              'turn_count': 2}]


def test_block_summary_counts_errors_fallbacks_caps_and_agreement(tmp_path):
    d = _block(tmp_path, LOG, DECISIONS)
    games = sm.summarize_block({'name': 'B', 'opponent': 'pomcp_ts_cr', 'games': 2}, d, 20261108, 1)
    g, missing = games
    assert g['finished'] and g['seat'] == 3 and g['won'] and g['our_vp'] == 10 and g['our_position'] == 1
    assert g['bridge_errors'] == 1 and g['fallbacks_by_type'] == {'actions': 1}
    assert g['mcts'] == {'searches': 2, 'cap_hits': 1, 'cap_overruns_1s': 0, 'ms_p50': 900, 'ms_p95': 10120,
                         'ms_max': 10120}
    assert g['stac_results_agree'] and g['player_trades_executed'] == 0
    assert g['seed_logged_matches'] is False  # logged 42 is not the derived seed
    assert not missing['finished']
    total = sm.aggregate(games, 2)
    assert total['truncated'] == 1 and total['bridge_errors'] == 1 and total['mcts_cap_hits'] == 1
    assert total['seed_mismatches'] == ['B_0', 'B_1']  # an unlogged seed is a mismatch (fail closed)
    assert total['force_ends_ours'] == 0 and total['force_ends'] == 1


def test_observability_violation_is_reported(tmp_path):
    d = _block(tmp_path, LOG.replace('fully_observable=false', 'fully_observable=true'), DECISIONS)
    games = sm.summarize_block({'name': 'B', 'opponent': 'jsettlers', 'games': 1}, d, None, 1)
    assert sm.aggregate(games, 1)['observability_violations'] == ['B_0']


def _receipt(tmp_path, tag, dice, decisions, winner=3):
    d = tmp_path / tag / 'B'
    d.mkdir(parents=True)
    (d / 'decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decisions))
    seq = sm.js.decision_sequences(decisions).get('B_0', [])
    game = {'game': 'B_0', 'block': 'B', 'finished': True, 'board_sha': 'b', 'deck': '[1]', 'first_player': 2,
            'seats': ['a', 'b', 'c', 'CatanAi'], 'seat': 3, 'dice': dice, 'dice_sha': sm.seq_sha(dice),
            'winner': winner, 'turns': 90, 'decision_seq_sha': sm.seq_sha(seq)}
    receipt = {'stac': {'seed': 1, 'patch_sha256': 'p'}, 'config': {'policy': 'x', 'seed': 1, 'blocks': [], 'agents': {}},
               'games_requested': 1, 'games': [game], 'artifacts': {'B': str(d)}}
    path = tmp_path / f'{tag}.json'
    path.write_text(json.dumps(receipt))
    return path


def test_compare_identical_and_divergent_runs(tmp_path):
    dec = [{'type': 'decision', 'game': 'B_0', 'decision': 'roll', 'turn_count': 1, 'cmd': 'roll'}]
    a = _receipt(tmp_path, 'a', ['0:1:2:6,2', '1:2:3:1,4'], dec)
    b = _receipt(tmp_path, 'b', ['0:1:2:6,2', '1:2:3:1,4'], dec)
    same = sm.compare_receipts(a, b)
    assert same['compatible'] and same['complete_games'] == 1 and same['end_to_end_identical'] == 1
    dec2 = dec + [{'type': 'decision', 'game': 'B_0', 'decision': 'actions', 'turn_count': 1, 'cmd': 'end_turn'}]
    c = _receipt(tmp_path, 'c', ['0:1:2:6,2', '1:3:0:1,4', '2:4:1:3,3'], dec2, winner=0)
    diff = sm.compare_receipts(a, c)
    assert diff['setup_and_dice_k_identical'] == 1 and diff['end_to_end_identical'] == 0
    assert diff['rows'][0]['first_decision_divergence']['index'] == 1


def test_compare_fails_closed_on_missing_evidence(tmp_path):
    dec = [{'type': 'decision', 'game': 'B_0', 'decision': 'roll', 'turn_count': 1, 'cmd': 'roll'}]
    a = _receipt(tmp_path, 'a', ['0:1:2:6,2'], dec)
    b = _receipt(tmp_path, 'b', [], dec)  # no dice logged
    result = sm.compare_receipts(a, b)
    assert result['incomplete_games'] == 1 and result['complete_games'] == 0
    unseeded = json.loads(a.read_text())
    unseeded['stac']['seed'] = None
    (tmp_path / 'u.json').write_text(json.dumps(unseeded))
    assert not sm.compare_receipts(tmp_path / 'u.json', tmp_path / 'u.json')['compatible']
    empty = _receipt(tmp_path, 'e', ['0:1:2:6,2'], [])
    assert sm.compare_receipts(a, empty)['incomplete_games'] == 1


def test_bridge_sources_include_the_reused_transport():
    names = {p.name for p in sb.bridge_sources()}
    assert {'CatanAiStacBrain.java', 'CatanAiStacFactory.java', 'StacStateJson.java',
            'PythonBridge.java', 'DecisionLog.java'} <= names


def test_classpath_puts_stac_classes_before_the_mcts_jar(tmp_path):
    cp = sb.classpath(tmp_path).split(':')
    assert cp[0].endswith('stac-classes') and cp[1].endswith('catanai-classes')
    assert cp.index(str(tmp_path / 'lib' / 'MCTS-1.0.jar')) > 0


HAVE_STAC = (Path(sb.java('javac')).exists() and (sb.CHECKOUT / '.git').exists() and sb.DEPS.is_dir()
             and (ROOT / SMOKE['stac']['patch']['file']).is_file())
needs_stac = pytest.mark.skipif(not HAVE_STAC, reason='needs the local JDK, StacSettlers checkout, deps and patch')


def test_patch_sha_mismatch_and_missing_patch_refused(tmp_path):
    with pytest.raises(sb.BuildIdentityError):
        sb.check_patch(None)
    bogus = tmp_path / 'p.patch'
    bogus.write_text('x')
    with pytest.raises(sb.BuildIdentityError, match='sha256'):
        sb.check_patch({'file': str(bogus), 'sha256': '0' * 64})


def test_existing_build_dir_refused(tmp_path):
    (tmp_path / 'build').mkdir()
    with pytest.raises(sb.BuildIdentityError, match='existing'):
        sb.stage_stac(tmp_path / 'build', SMOKE['stac']['patch'])


@needs_stac
def test_build_reproduces_pin_plus_patch_and_detects_tampering(tmp_path):
    manifest = sb.build_manifest(tmp_path / 'build', SMOKE['stac']['patch'])
    assert manifest['stac']['commit'] == sb.STAC_COMMIT
    assert manifest['stac']['patch_sha256'] == SMOKE['stac']['patch']['sha256']
    assert manifest['stac']['class_files'] > 600
    assert 'catanai/stac/CatanAiStacBrain.class' in manifest['bridge_java']['classes']
    sb.verify_loaded(manifest)
    victim = tmp_path / 'build' / 'stac-classes' / 'soc' / 'game' / 'CatanAiSeed.class'
    victim.write_bytes(victim.read_bytes() + b'\0')
    with pytest.raises(sb.BuildIdentityError):
        sb.verify_loaded(manifest)
    with pytest.raises(sb.BuildIdentityError):
        sb.verify_stac(tmp_path / 'build', SMOKE['stac']['patch'])


def _chat_config(jsettlers_block=True):
    config = copy.deepcopy(SMOKE)
    config['stac'].update(player_trading='chat', chat_negotiation=True, no_player_trades=False)
    config['agents']['pomcp_ts_cr']['flags'].remove('NO_TRADES')
    config['agents']['stac_negotiator'] = {'source': 'STAC StacRobotBrain'}
    if not jsettlers_block:
        config['blocks'] = [config['blocks'][0], {'name': 'Neg', 'opponent': 'stac_negotiator', 'games': 1}]
    return config


def test_chat_trading_mode_settings():
    config = _chat_config(jsettlers_block=False)
    stac = sm.check_config(config)
    assert sm.trading_mode(stac) == 'chat' and sm.trading_mode(SMOKE['stac']) == 'off'
    lines = sm.simulation_config(config['blocks'][0], config['agents'], 'chat').splitlines()
    assert 'ChatNegotiation=true' in lines and 'Trades=true' in lines
    assert not any('NO_TRADES' in line for line in lines)
    props = sm.jvm_props(stac)
    assert not any('noPlayerTrades' in p for p in props) and '-Dstac.catanai.forceEndTurnSec=30' in props
    logged = {'fully_observable': 'false', 'observable_vp': 'false', 'chat_negotiations': 'true',
              'no_player_trades': 'false'}
    assert sm.observability_ok({'observability': logged}, 'chat')
    assert not sm.observability_ok({'observability': logged}, 'off')
    assert not sm.observability_ok({'observability': {**logged, 'fully_observable': 'true'}}, 'chat')
    neg = sm.simulation_config(config['blocks'][1], config['agents'], 'chat').splitlines()
    assert 'Agent=3,StacNeg,stac' in neg


def test_negotiator_needs_chat_mode():
    config = copy.deepcopy(SMOKE)
    config['agents']['stac_negotiator'] = {'source': 'x'}
    config['blocks'].append({'name': 'Neg', 'opponent': 'stac_negotiator', 'games': 1})
    with pytest.raises(sm.RunRefused, match='stac_negotiator cannot play with player_trading off'):
        sm.check_config(config)


@pytest.mark.parametrize('mutate, message', [
    (lambda c: c['stac'].update(player_trading='maybe'), 'player_trading'),
    (lambda c: c['stac'].update(chat_negotiation=False), 'chat_negotiation'),
    (lambda c: c['stac'].update(no_player_trades=True), 'no_player_trades'),
    (lambda c: c['agents']['pomcp_ts_cr']['flags'].append('NO_TRADES'), 'NO_TRADES'),
    (lambda c: c['stac'].update(observable_vp=True), 'observable_vp'),
    (lambda c: None, 'opponent jsettlers cannot play with player_trading chat'),
])
def test_chat_trading_refusals(mutate, message):
    config = _chat_config()
    if message.startswith('opponent'):
        mutate(config)
    else:
        config['blocks'] = config['blocks'][:1]
        mutate(config)
    with pytest.raises(sm.RunRefused, match=message):
        sm.check_config(config)


def test_chat_trade_events_are_counted(tmp_path):
    events = [{'type': 'chat_trade', 'game': 'B_0', 'event': e, 'seat': 3}
              for e in ('offer_sent', 'offer_sent', 'offer_accepted', 'offer_rejected', 'answered_accept')]
    d = _block(tmp_path, LOG, DECISIONS + events)
    games = sm.summarize_block({'name': 'B', 'opponent': 'pomcp_ts_cr', 'games': 1}, d, None, 1)
    assert games[0]['chat_trades'] == {'offer_sent': 2, 'offer_accepted': 1, 'offer_rejected': 1,
                                       'answered_accept': 1}
    assert sm.aggregate(games, 1, 'chat')['chat_trades']['offer_sent'] == 2
