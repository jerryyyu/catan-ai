from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from catan_ai.api import app
from catan_ai import rooms, pilot_store
from catan_ai.bots import make_policy
from catan_ai.observation import Observation


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(rooms.secrets, 'randbits', lambda _: 1469000)
    rooms.rooms.clear()
    with TestClient(app) as client:
        yield client
    rooms.rooms.clear()


def create(client, bot_seat=0):
    result = client.post('/api/rooms', json={'bot_seat': bot_seat}).json()
    room_id = result['room_id']
    headers = {int(s): {'Authorization': 'Bearer ' + token} for s, token in result['invitations'].items()}
    return room_id, headers


def join_all(client, room_id, headers):
    for seat, header in headers.items():
        response = client.post(f'/api/rooms/{room_id}/join', headers=header,
                               json={'name': f'Test {seat}', 'completed_matches': 50,
                                     'platform': 'synthetic test fixture', 'consent': True})
        assert response.status_code == 200, response.text


def test_invitation_auth_lobby_and_private_views(client):
    room_id, headers = create(client)
    for endpoint in (f'/api/rooms/{room_id}', f'/api/rooms/{room_id}/replay'):
        assert client.get(endpoint).status_code == 403
    first = next(iter(headers))
    state = client.get(f'/api/rooms/{room_id}', headers=headers[first]).json()
    assert state['ready'] is False and state['state']['legal_actions'] == []
    assert client.post(f'/api/rooms/{room_id}/actions', headers=headers[first],
                       json={'revision': 0, 'action': {'type': 'roll'}}).status_code == 409
    join_all(client, room_id, headers)
    for seat, header in headers.items():
        state = client.get(f'/api/rooms/{room_id}', headers=header).json()
        assert state['ready'] and state['seat'] == seat
        assert state['revision'] == 2
        for player in state['state']['players']:
            assert (player['resources'] is not None) == (player['seat'] == seat)
            assert (player['development'] is not None) == (player['seat'] == seat)
        assert 'participants' not in state and 'seat_hashes' not in state
    raw = pilot_store.load('room_' + room_id)
    assert all(header['Authorization'][7:] not in str(raw) for header in headers.values())


def test_action_revision_wrong_seat_and_restart_replay(client):
    room_id, headers = create(client)
    join_all(client, room_id, headers)
    state = client.get(f'/api/rooms/{room_id}', headers=headers[1]).json()
    action = state['state']['legal_actions'][0]
    body = {'revision': state['revision'], 'action': action}
    assert client.post(f'/api/rooms/{room_id}/actions', headers=headers[2], json=body).status_code == 400
    result = client.post(f'/api/rooms/{room_id}/actions', headers=headers[1], json=body)
    assert result.status_code == 200
    assert client.post(f'/api/rooms/{room_id}/actions', headers=headers[1], json=body).status_code == 409
    expected = deepcopy(result.json())
    raw = pilot_store.load('room_' + room_id)
    replay = client.get(f'/api/rooms/{room_id}/replay?step=0', headers=headers[1]).json()
    assert replay['step'] == 0 and not replay['events']
    assert 'seed' not in replay and 'trace' not in replay
    assert pilot_store.load('room_' + room_id) == raw
    rooms.rooms.clear()
    restored = client.get(f'/api/rooms/{room_id}', headers=headers[1])
    assert restored.status_code == 200, restored.text
    assert restored.json() == expected


def test_complete_three_client_match_is_fixture_not_human_evidence(client):
    room_id, headers = create(client, bot_seat=2)
    join_all(client, room_id, headers)
    drivers = {seat: make_policy('heuristic', 7 + seat) for seat in headers}
    for _ in range(5000):
        room = rooms.rooms[room_id]
        if room.game.winner is not None:
            break
        seat = room.game.current
        assert seat in headers
        state = client.get(f'/api/rooms/{room_id}', headers=headers[seat]).json()
        action = drivers[seat].choose(Observation.for_policy(room.game, seat, drivers[seat]), seat)
        result = client.post(f'/api/rooms/{room_id}/actions', headers=headers[seat],
                             json={'revision': state['revision'], 'action': action})
        assert result.status_code == 200, result.text
    else:
        pytest.fail('synthetic room match did not finish')
    expected = json.loads(json.dumps(room.game.public_state(0)))
    rooms.rooms.clear()
    restored = client.get(f'/api/rooms/{room_id}', headers=headers[0])
    assert restored.status_code == 200, restored.text
    assert restored.json()['state'] == expected
    replay = client.get(f'/api/rooms/{room_id}/replay', headers=headers[0]).json()
    assert replay['state'] == expected
