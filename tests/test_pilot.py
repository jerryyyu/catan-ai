from fastapi.testclient import TestClient
import json
import pytest

from catan_ai.api import app, sessions
from catan_ai import pilot_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('CATAN_PILOT_DIR', str(tmp_path))
    sessions.clear()
    return TestClient(app)


@pytest.mark.parametrize('human_seat', range(4))
def test_restart_replays_bot_rng_and_continues_identically(client, human_seat):
    created = client.post('/api/games', json={'seed': 1432000, 'human_seat': human_seat}).json()
    assert created['seat'] == human_seat
    assert created['state']['current'] == human_seat
    game_id = created['game_id']
    url = f'/api/games/{game_id}'
    state = created['state']
    for _ in range(15):
        response = client.post(url + '/actions', json={'action': state['legal_actions'][0]})
        assert response.status_code == 200
        state = response.json()
    original = sessions.pop(game_id)
    resumed = client.get(url)
    assert resumed.status_code == 200
    assert resumed.json() == state
    rebuilt = sessions[game_id]
    action = state['legal_actions'][0]
    actual = client.post(url + '/actions', json={'action': action}).json()
    sessions[game_id] = original
    expected = client.post(url + '/actions', json={'action': action}).json()
    assert actual == expected
    assert rebuilt.game.rng.getstate() == original.game.rng.getstate()
    assert all(rebuilt.bots[seat].rng.getstate() == original.bots[seat].rng.getstate()
               for seat in range(4) if seat != human_seat)
    replay = client.get(url + '/replay').json()
    assert replay['human_seat'] == human_seat
    assert replay['events'] == original.game.observed_events(human_seat)
    for seat, player in enumerate(replay['state']['players']):
        assert (player['resources'] is not None) == (seat == human_seat)
        assert (player['development'] is not None) == (seat == human_seat)


@pytest.mark.parametrize('human_seat', [-1, 4, True, '1'])
def test_invalid_seat_rejected_before_creating_session(client, human_seat):
    assert client.post('/api/games', json={'human_seat': human_seat}).status_code == 422
    assert not sessions


def test_export_is_player_view_and_invalid_action_does_not_enter_record(client):
    created = client.post('/api/games', json={'seed': 1432001}).json()
    game_id = created['game_id']
    url = f'/api/games/{game_id}'
    assert client.post(url + '/actions', json={'action': {'type': 'roll'}}).status_code == 400
    assert pilot_store.load(game_id)['trace'] == []
    result = client.get(url + '/replay').json()
    assert 'seed' not in result and 'trace' not in result
    assert result['events'] == sessions[game_id].game.observed_events(0)
    assert all(p['resources'] is None and p['development'] is None for p in result['state']['players'][1:])
    assert result['completed'] is False


def test_changed_source_refuses_recovery_and_preserves_record(client):
    game_id = client.post('/api/games', json={'seed': 1432002}).json()['game_id']
    record = pilot_store.load(game_id)
    record['source']['changed'] = 'not-current'
    pilot_store.save(game_id, record)
    before = pilot_store.record_path(game_id).read_bytes()
    sessions.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 409
    assert pilot_store.record_path(game_id).read_bytes() == before


def test_record_paths_cannot_escape_store():
    with pytest.raises(FileNotFoundError):
        pilot_store.record_path('../outside')


@pytest.mark.parametrize('human_seat', range(4))
def test_replay_steps_reconstruct_without_mutating_live_game(client, human_seat):
    created = client.post('/api/games', json={'seed': 1452000, 'human_seat': human_seat}).json()
    game_id = created['game_id']
    url = f'/api/games/{game_id}'
    session = sessions[game_id]
    initial_state = created['state']
    initial_step = len(session.trace)
    state = initial_state
    for _ in range(8):
        state = client.post(url + '/actions', json={'action': state['legal_actions'][0]}).json()
    before_rng = session.game.rng.getstate()
    before_record = pilot_store.record_path(game_id).read_bytes()
    replay = client.get(url + f'/replay?step={initial_step}').json()
    assert replay['state'] == initial_state
    assert len(replay['events']) == initial_step
    assert len(replay['timings']) == initial_step
    assert 'seed' not in replay and 'trace' not in replay
    for seat, player in enumerate(replay['state']['players']):
        assert (player['resources'] is not None) == (seat == human_seat)
    assert client.get(url + '/replay?step=0').json()['events'] == []
    assert client.get(url + '/replay?step=-1').status_code == 422
    assert client.get(url + f'/replay?step={len(session.trace) + 1}').status_code == 422
    assert json.loads(json.dumps(session.game.public_state(human_seat))) == state
    assert session.game.rng.getstate() == before_rng
    assert pilot_store.record_path(game_id).read_bytes() == before_record


@pytest.fixture
def parameter_model(tmp_path, monkeypatch):
    from catan_ai.experiments import source_files
    from catan_ai.parameter_planner import DEFAULT
    path = tmp_path / 'synthetic-model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))
    return path


@pytest.mark.parametrize('human_seat', range(4))
def test_fitted_solo_restart_privacy_and_checkpoint_identity(client, parameter_model, human_seat):
    import hashlib
    created = client.post('/api/games', json={'seed': 1623000, 'human_seat': human_seat,
                                             'opponent': 'planner_parameter'})
    assert created.status_code == 200, created.text
    data = created.json()
    game_id = data['game_id']
    url = f'/api/games/{game_id}'
    record = pilot_store.load(game_id)
    assert record['model_sha256'] == hashlib.sha256(parameter_model.read_bytes()).hexdigest()
    assert 'model_sha256' not in data
    state = data['state']
    for _ in range(3):
        played = client.post(url + '/actions', json={'action': state['legal_actions'][0]})
        assert played.status_code == 200, played.text
        state = played.json()
    original = sessions.pop(game_id)
    restored = client.get(url)
    assert restored.status_code == 200 and restored.json() == state
    assert sessions[game_id].game.rng.getstate() == original.game.rng.getstate()
    assert all(sessions[game_id].bots[s].rng.getstate() == original.bots[s].rng.getstate()
               for s in original.bots)
    for payload in (restored.json(), client.get(url + '/replay').json()):
        assert 'model_sha256' not in payload and 'source' not in payload
        view = payload.get('state', payload)
        for seat, player in enumerate(view['players']):
            assert (player['resources'] is not None) == (seat == human_seat)
            assert (player['development'] is not None) == (seat == human_seat)
    before = pilot_store.record_path(game_id).read_bytes()
    model = json.loads(parameter_model.read_text())
    model['lineage'] = 'changed metadata, unchanged weights'
    parameter_model.write_text(json.dumps(model))
    sessions.clear()
    assert client.get(url).status_code == 409
    assert pilot_store.record_path(game_id).read_bytes() == before


def test_fitted_solo_requires_checkpoint_before_saving(client, monkeypatch, tmp_path):
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(tmp_path / 'missing.json'))
    assert client.post('/api/games', json={'opponent': 'planner_parameter'}).status_code == 503
    assert not sessions


def test_fitted_solo_missing_saved_identity_fails_closed(client, parameter_model):
    data = client.post('/api/games', json={'seed': 1623000, 'opponent': 'planner_parameter'}).json()
    record = pilot_store.load(data['game_id'])
    del record['model_sha256']
    pilot_store.save(data['game_id'], record)
    sessions.clear()
    assert client.get('/api/games/' + data['game_id']).status_code == 409
