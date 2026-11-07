"""Decision bridge: served-policy answers translated to JSettlers commands, fallbacks, server loop."""
import copy
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'jsettlers'))

from jsbridge.decide import POLICY, Bridge, translate  # noqa: E402
from jsbridge.server import serve  # noqa: E402
from jsbridge.state import reconstruct  # noqa: E402

RECORDS = [json.loads(line) for line in
           (ROOT / 'tools' / 'jsettlers' / 'fixtures' / 'snapshots.jsonl').read_text().splitlines()]
CMDS = {'setup_settlement': {'settlement'}, 'setup_road': {'road'}, 'roll': {'roll', 'play_dev'},
        'actions': {'settlement', 'road', 'city', 'buy_dev', 'bank', 'play_dev', 'offer', 'end_turn'},
        'free_roads': {'road'}, 'robber': {'robber'}, 'steal': {'steal'}, 'discard': {'discard'},
        'trade_response': {'accept', 'reject'}}


def test_served_policy_is_the_bridge_default():
    assert POLICY == 'planner_offer_accept'


@pytest.mark.parametrize('record', RECORDS, ids=[f'{i}-{r["request"]["decision"]}' for i, r in enumerate(RECORDS)])
def test_decision_is_a_jsettlers_legal_command(record):
    snap = record['request']
    reply = Bridge(seed=7).decide(copy.deepcopy(snap))
    cmd = reply['cmd']
    assert cmd in CMDS[snap['decision']], reply
    legal = snap['js_legal']
    if cmd in ('settlement',):
        assert reply['node'] in legal['settlements']
    if cmd == 'city':
        assert reply['node'] in legal['cities']
    if cmd == 'road':
        assert reply['edge'] in legal['roads']
    if cmd == 'robber':
        assert reply['hex'] in legal['robber_hexes']
    if cmd == 'steal':
        assert reply['victim'] in legal['victims']
    if cmd == 'discard':
        assert sum(reply['resources'].values()) == snap['discard_count']
        for code, n in reply['resources'].items():
            assert snap['own']['resources'][code] >= n
    if cmd == 'bank':
        (give, n), = reply['give'].items()
        assert n == legal['rates'][give] and snap['own']['resources'][give] >= n
        assert sum(reply['get'].values()) == 1
    if cmd == 'offer':
        assert reply['to'] != snap['seat'] and reply['give'] and reply['get']
        assert not set(reply['give']) & set(reply['get'])
    if cmd == 'play_dev':
        if reply['card'] == 'year_of_plenty':
            assert sum(reply['pick'].values()) == 2
        if reply['card'] == 'monopoly':
            assert reply['resource'] in range(1, 6)


def _snap(decision, action_type=None):
    return copy.deepcopy(next(r['request'] for r in RECORDS if r['request']['decision'] == decision
                              and (action_type is None or r['reply']['action']['type'] == action_type)))


def test_target_illegal_in_jsettlers_becomes_a_counted_fallback():
    snap = _snap('setup_settlement')
    snap['js_legal']['settlements'] = []
    reply = Bridge().decide(snap)
    assert reply['cmd'] == 'fallback' and 'not legal in JSettlers' in reply['reason']


def test_rate_disagreement_is_a_fallback_not_a_silent_trade():
    snap = _snap('actions', 'bank_trade')
    game, obs, ctx = reconstruct(snap)
    action = next(a for a in obs.actions if a['type'] == 'bank_trade')
    give = {'brick': '1', 'ore': '2', 'wool': '3', 'grain': '4', 'lumber': '5'}[action['give']]
    snap['js_legal']['rates'][give] = 1
    assert translate(action, snap, game, ctx)['cmd'] == 'fallback'


def test_card_translations():
    snap = _snap('actions')
    game, obs, ctx = reconstruct(snap)
    mono = translate({'type': 'play_development', 'card': 'monopoly', 'resource': 'ore'}, snap, game, ctx)
    assert (mono['cmd'], mono['card'], mono['card_type'], mono['resource']) == ('play_dev', 'monopoly', 3, 2)
    yop = translate({'type': 'play_development', 'card': 'year_of_plenty', 'resources': ['wool', 'wool']},
                    snap, game, ctx)
    assert yop['pick'] == {'3': 2} and yop['card_type'] == 2
    knight = translate({'type': 'play_development', 'card': 'knight'}, snap, game, ctx)
    assert knight['card_type'] == 9
    rb = translate({'type': 'play_development', 'card': 'road_building'}, snap, game, ctx)
    assert rb['card_type'] == 1
    # The engine's one-card Year of Plenty (bank nearly empty) has no JSettlers equivalent.
    single = translate({'type': 'play_development', 'card': 'year_of_plenty', 'resources': ['wool']},
                       snap, game, ctx)
    assert single['cmd'] == 'fallback'


def test_offer_goes_to_one_jsettlers_seat():
    snap = _snap('actions')
    game, obs, ctx = reconstruct(snap)
    target = (ctx['seat'] + 1) % 4
    reply = translate({'type': 'offer_trade', 'recipient': target, 'give': {'wool': 1}, 'receive': {'ore': 1}},
                      snap, game, ctx)
    assert reply == {'action': reply['action'], 'cmd': 'offer', 'to': ctx['seats'].js(target),
                     'give': {'3': 1}, 'get': {'2': 1}}


def test_server_answers_every_line_and_survives_errors():
    good = dict(_snap('roll'), kind='decide', id=3)
    broken = dict(_snap('roll'), kind='decide', id=4)
    broken['board'] = {'hexes': []}
    lines = [json.dumps({'kind': 'ping', 'id': 1}), 'not json', json.dumps(good), json.dumps(broken),
             json.dumps({'kind': 'end_game', 'game': good['game'], 'id': 5}), json.dumps({'kind': 'x', 'id': 6})]
    out, log = io.StringIO(), io.StringIO()
    serve(io.StringIO('\n'.join(lines) + '\n'), out, Bridge(), log)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r['cmd'] for r in replies] == ['pong', 'error', replies[2]['cmd'], 'error', 'ok', 'error']
    assert replies[2]['cmd'] in ('roll', 'play_dev') and replies[2]['id'] == 3
    assert replies[3]['id'] == 4 and 'error' in replies[3]
    assert len(log.getvalue().splitlines()) == 6
