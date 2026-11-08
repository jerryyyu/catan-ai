"""Recorded JSettlers bot-side snapshots -> engine Observation, and the hidden-information boundary."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'jsettlers'))

from catan_ai.card_planner import resource_estimates  # noqa: E402
from catan_ai.engine import RESOURCES  # noqa: E402
from jsbridge.board import CODE_RESOURCE  # noqa: E402
from jsbridge.decide import legal_mismatches  # noqa: E402
from jsbridge.state import (OPPONENT_KEYS, SnapshotError, check_hidden_boundary,  # noqa: E402
                            reconstruct)

RECORDS = [json.loads(line) for line in
           (ROOT / 'tools' / 'jsettlers' / 'fixtures' / 'snapshots.jsonl').read_text().splitlines()]
PHASE = {'setup_settlement': 'setup_settlement', 'setup_road': 'setup_road', 'roll': 'roll',
         'actions': 'actions', 'free_roads': 'free_roads', 'robber': 'robber', 'steal': 'steal',
         'discard': 'discard', 'trade_response': 'trade_response'}


def ids(records):
    return [f'{i}-{r["request"]["decision"]}' for i, r in enumerate(records)]


def test_fixture_covers_every_decision_type():
    assert {r['request']['decision'] for r in RECORDS} == set(PHASE)


@pytest.mark.parametrize('record', RECORDS, ids=ids(RECORDS))
def test_snapshot_reconstructs_with_matching_legal_sets(record):
    snap = record['request']
    game, obs, ctx = reconstruct(snap)
    seats = ctx['seats']
    assert obs.phase == PHASE[snap['decision']]
    assert obs.seat == seats.engine(snap['seat'])
    assert seats.engine(snap['first_player']) == 0
    # Engine legal actions agree with JSettlers' legal sets for every checkable decision type
    # (placements, cities, port rates, robber hexes, victims).
    assert legal_mismatches(snap, obs, ctx) == []
    # Own hand and pieces are exact.
    me = obs.players[obs.seat]
    assert {r: n for r, n in me.resources.items() if n} == \
        {CODE_RESOURCE[int(k)]: v for k, v in snap['own']['resources'].items()}
    js_me = next(p for p in snap['players'] if p['pn'] == snap['seat'])
    m = ctx['board_map']
    assert me.settlements == {m.node_to_vertex[n] for n in js_me['settlements']}
    assert me.roads == {m.edge_to_engine[e] for e in js_me['roads']}
    # The action the bridge chose from this snapshot was engine-legal here.
    action = record['reply'].get('action')
    if action and action['type'] not in ('discard', 'offer_trade'):
        assert action in obs.actions
    if snap['decision'] == 'steal':
        assert sorted(seats.js(a['victim']) for a in obs.actions) == sorted(snap['js_legal']['victims'])
    if snap['decision'] == 'trade_response':
        assert obs.trade_offer['owner'] == seats.engine(snap['offer']['from'])


@pytest.mark.parametrize('record', RECORDS, ids=ids(RECORDS))
def test_hidden_information_boundary(record):
    snap = record['request']
    check_hidden_boundary(snap)
    me = snap['seat']
    for p in snap['players']:
        # Only public facts per seat: pieces, counts, and JSettlers' known/unknown split.
        assert set(p) <= OPPONENT_KEYS
        assert 'resources' not in p and 'dev' not in p
    # Our exact hand and card list appear once, in the own block.
    assert set(snap['own']) == {'resources', 'dev'}
    _, obs, ctx = reconstruct(snap)
    for seat, view in enumerate(obs.players):
        if seat == obs.seat:
            continue
        assert view.resources is None and view.development is None and view.development_turns is None
    # The synthetic history folds to exactly JSettlers' bot-side tracking: known + unknown / 5.
    estimates = resource_estimates(obs.events)
    for p in snap['players']:
        seat = ctx['seats'].engine(p['pn'])
        for r in RESOURCES:
            code = next(k for k, v in CODE_RESOURCE.items() if v == r)
            expected = p['known'].get(str(code), 0) + p['unknown'] / 5
            assert estimates[seat][r] == pytest.approx(expected)
        if p['pn'] == me:
            assert p['unknown'] == 0


def test_boundary_rejects_leaked_opponent_hand():
    snap = copy.deepcopy(next(r['request'] for r in RECORDS if any(p['unknown'] for p in r['request']['players'])))
    opponent = next(p for p in snap['players'] if p['pn'] != snap['seat'])
    opponent['resources'] = {'1': 3}
    with pytest.raises(SnapshotError):
        check_hidden_boundary(snap)
    with pytest.raises(SnapshotError):
        reconstruct(snap)


def test_boundary_rejects_opponent_card_list_and_inconsistent_counts():
    snap = copy.deepcopy(RECORDS[0]['request'])
    opponent = next(p for p in snap['players'] if p['pn'] != snap['seat'])
    leaked = copy.deepcopy(snap)
    next(p for p in leaked['players'] if p['pn'] == opponent['pn'])['dev'] = [{'type': 9}]
    with pytest.raises(SnapshotError):
        check_hidden_boundary(leaked)
    wrong = copy.deepcopy(snap)
    next(p for p in wrong['players'] if p['pn'] == opponent['pn'])['total'] += 1
    with pytest.raises(SnapshotError):
        check_hidden_boundary(wrong)
