"""StacSettlers bridge: recorded STAC snapshots -> engine state, hidden-information boundary, STAC adapter.

The Python side is the JSettlers bridge (``tools/jsettlers/jsbridge``) reused unchanged; these tests
run it on snapshots recorded from real StacSettlers games (``tools/stac/fixtures/snapshots.jsonl``,
built by ``catanai.stac.StacStateJson``) and test the STAC adapter (``tools/stac/stacbridge``).
"""
import copy
import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'jsettlers'))
sys.path.insert(0, str(ROOT / 'tools' / 'stac'))

from catan_ai.card_planner import resource_estimates  # noqa: E402
from catan_ai.engine import BACKEND, RESOURCES  # noqa: E402
from catan_ai.experiments import source_files  # noqa: E402
from catan_ai.parameter_planner import DEFAULT  # noqa: E402
from jsbridge.board import CODE_RESOURCE, check_base_board, engine_board  # noqa: E402
from jsbridge.decide import legal_mismatches  # noqa: E402
from jsbridge.server import serve  # noqa: E402
from jsbridge.state import OPPONENT_KEYS, SnapshotError, check_hidden_boundary, reconstruct  # noqa: E402
from stacbridge.decide import StacBridge, TradingSwitchMissing  # noqa: E402

RECORDS = [json.loads(line) for line in
           (ROOT / 'tools' / 'stac' / 'fixtures' / 'snapshots.jsonl').read_text().splitlines()]
PHASE = {'setup_settlement': 'setup_settlement', 'setup_road': 'setup_road', 'roll': 'roll',
         'actions': 'actions', 'free_roads': 'free_roads', 'robber': 'robber', 'steal': 'steal',
         'discard': 'discard'}
CMDS = {'setup_settlement': {'settlement'}, 'setup_road': {'road'}, 'roll': {'roll', 'play_dev'},
        'actions': {'settlement', 'road', 'city', 'buy_dev', 'bank', 'play_dev', 'end_turn'},
        'free_roads': {'road'}, 'robber': {'robber'}, 'steal': {'steal'}, 'discard': {'discard'}}


def ids(records):
    return [f'{i}-{r["request"]["decision"]}' for i, r in enumerate(records)]


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    """Bind the planner to the active engine backend (as in tests/test_jsettlers_decide.py)."""
    if BACKEND == 'python':
        monkeypatch.delenv('CATAN_PARAMETER_MODEL', raising=False)
    else:
        path = tmp_path / 'parameters.json'
        path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
        monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def test_fixture_covers_every_stac_decision_type():
    # Player trading is off in STAC matches, so trade_response never occurs.
    assert {r['request']['decision'] for r in RECORDS} == set(PHASE)
    assert len({r['request']['game'] for r in RECORDS}) >= 2


@pytest.mark.parametrize('record', RECORDS, ids=ids(RECORDS))
def test_stac_board_maps_onto_a_standard_engine_board(record):
    board, bmap = engine_board(record['request']['board'])
    check_base_board(board)
    assert len(bmap.hex_to_tile) == 19 and len(bmap.node_to_vertex) == 54 and len(bmap.edge_to_engine) == 72
    # STAC 1.1.16 desert (0) is translated to the JSettlers2 code the mapping expects.
    assert sum(h['type'] == 6 for h in record['request']['board']['hexes']) == 1


@pytest.mark.parametrize('record', RECORDS, ids=ids(RECORDS))
def test_stac_snapshot_reconstructs_with_matching_legal_sets(record):
    snap = record['request']
    game, obs, ctx = reconstruct(snap)
    seats = ctx['seats']
    assert obs.phase == PHASE[snap['decision']]
    assert obs.seat == seats.engine(snap['seat'])
    assert legal_mismatches(snap, obs, ctx) == []
    me = obs.players[obs.seat]
    assert {r: n for r, n in me.resources.items() if n} == \
        {CODE_RESOURCE[int(k)]: v for k, v in snap['own']['resources'].items()}
    action = record['reply'].get('action')
    if action and action['type'] != 'discard':
        assert action in obs.actions
    if snap['decision'] == 'steal':
        assert sorted(seats.js(a['victim']) for a in obs.actions) == sorted(snap['js_legal']['victims'])


@pytest.mark.parametrize('record', RECORDS, ids=ids(RECORDS))
def test_stac_hidden_information_boundary(record):
    snap = record['request']
    check_hidden_boundary(snap)
    for p in snap['players']:
        assert set(p) <= OPPONENT_KEYS
    assert set(snap['own']) == {'resources', 'dev'}
    _, obs, ctx = reconstruct(snap)
    for seat, view in enumerate(obs.players):
        if seat != obs.seat:
            assert view.resources is None and view.development is None and view.development_turns is None
    estimates = resource_estimates(obs.events)
    for p in snap['players']:
        seat = ctx['seats'].engine(p['pn'])
        for r in RESOURCES:
            code = next(k for k, v in CODE_RESOURCE.items() if v == r)
            assert estimates[seat][r] == pytest.approx(p['known'].get(str(code), 0) + p['unknown'] / 5)
        if p['pn'] == snap['seat']:
            assert p['unknown'] == 0


def test_fixtures_include_hidden_opponent_cards():
    # The boundary is exercised: some opponent hands are partly unknown to our seat (steals between others).
    assert any(p['unknown'] for r in RECORDS for p in r['request']['players'] if p['pn'] != r['request']['seat'])
    assert any(p['dev_count'] for r in RECORDS for p in r['request']['players'] if p['pn'] != r['request']['seat'])


def test_java_serializer_emits_only_public_per_player_keys():
    """StacStateJson writes per-player keys only from the boundary's allowed set."""
    src = (ROOT / 'tools' / 'stac' / 'java' / 'catanai' / 'stac' / 'StacStateJson.java').read_text()
    body = src.split('final JsonArray players = new JsonArray();', 1)[1].split('s.add("players", players);', 1)[0]
    keys = set(re.findall(r'p\.(?:add|addProperty)\("(\w+)"', body))
    assert keys == OPPONENT_KEYS
    own = src.split('final JsonObject own = new JsonObject();', 1)[1].split('s.add("own", own);', 1)[0]
    assert set(re.findall(r'own\.add\("(\w+)"', own)) == {'resources', 'dev'}


def test_boundary_rejects_leaked_opponent_hand():
    snap = copy.deepcopy(next(r['request'] for r in RECORDS
                              if any(p['unknown'] for p in r['request']['players'])))
    opponent = next(p for p in snap['players'] if p['pn'] != snap['seat'])
    opponent['resources'] = {'1': 3}
    with pytest.raises(SnapshotError):
        reconstruct(snap)


@pytest.mark.parametrize('record', RECORDS, ids=ids(RECORDS))
def test_stac_bridge_answers_with_a_legal_command(record):
    snap = record['request']
    reply = StacBridge(seed=7).decide(copy.deepcopy(snap))
    assert reply['cmd'] in CMDS[snap['decision']], reply
    legal = snap['js_legal']
    if reply['cmd'] == 'settlement':
        assert reply['node'] in legal['settlements']
    if reply['cmd'] == 'city':
        assert reply['node'] in legal['cities']
    if reply['cmd'] == 'road':
        assert reply['edge'] in legal['roads']
    if reply['cmd'] == 'robber':
        assert reply['hex'] in legal['robber_hexes']


def test_stac_bridge_disables_player_offers():
    bridge = StacBridge(seed=1)
    policy = bridge.policy_for('G_0')
    assert policy.offers is False
    assert bridge.policy_for('G_0') is policy


def test_offer_reaching_translation_becomes_a_counted_fallback(monkeypatch):
    snap = copy.deepcopy(next(r['request'] for r in RECORDS if r['request']['decision'] == 'actions'))
    import jsbridge.decide as jd
    monkeypatch.setattr(jd.Bridge, 'decide', lambda self, s: {'cmd': 'offer', 'action': {'type': 'offer_trade'},
                                                              'think_ms': 1.0})
    bridge = StacBridge(seed=1)
    reply = bridge.decide(snap)
    assert reply['cmd'] == 'fallback' and reply['suppressed_offer'] and 'disabled' in reply['reason']
    assert bridge.offers_suppressed == 1


def test_policy_without_an_offers_switch_is_refused(monkeypatch):
    import jsbridge.decide as jd
    monkeypatch.setattr(jd, 'make_policy', lambda name, seed: object())
    with pytest.raises(TradingSwitchMissing):
        StacBridge(seed=1).policy_for('G_0')


def test_non_ten_point_game_refused():
    snap = copy.deepcopy(RECORDS[0]['request'])
    snap['vp_to_win'] = 12
    with pytest.raises(SnapshotError):
        StacBridge(seed=1).decide(snap)


def test_server_loop_reports_errors_and_keeps_serving():
    snap = copy.deepcopy(RECORDS[0]['request'])
    bad = copy.deepcopy(snap)
    bad['vp_to_win'] = 12
    lines = [json.dumps({**bad, 'kind': 'decide', 'id': 1}), json.dumps({**snap, 'kind': 'decide', 'id': 2}),
             json.dumps({'kind': 'end_game', 'game': snap['game'], 'id': 3})]
    out = io.StringIO()
    serve(io.StringIO('\n'.join(lines) + '\n'), out, StacBridge(seed=1))
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r['id'] for r in replies] == [1, 2, 3]
    assert replies[0]['cmd'] == 'error' and replies[1]['cmd'] in CMDS[snap['decision']]
    assert replies[2]['cmd'] == 'ok'


def test_player_trades_mode_keeps_offers(monkeypatch):
    snap = copy.deepcopy(next(r['request'] for r in RECORDS if r['request']['decision'] == 'actions'))
    import jsbridge.decide as jd
    offer = {'cmd': 'offer', 'to': 1, 'give': {'1': 1}, 'get': {'4': 1}, 'action': {'type': 'offer_trade'},
             'think_ms': 1.0}
    monkeypatch.setattr(jd.Bridge, 'decide', lambda self, s: dict(offer))
    bridge = StacBridge(seed=1, player_trades=True)
    assert bridge.decide(snap) == offer and bridge.offers_suppressed == 0
    monkeypatch.undo()
    policy = StacBridge(seed=1, player_trades=True).policy_for('G_0')
    assert getattr(policy, 'offers', True) is not False
