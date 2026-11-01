from fastapi.testclient import TestClient

from catan_ai.api import app, sessions


def test_create_and_play_action_without_leaking_hidden_resources():
    sessions.clear()
    client = TestClient(app)
    response = client.post("/api/games", json={"name": "Player", "seed": 123})
    assert response.status_code == 200
    created = response.json()
    state = created["state"]
    assert state["current"] == 0
    assert state["players"][1]["resources"] is None

    action = state["legal_actions"][0]
    played = client.post(f"/api/games/{created['game_id']}/actions", json={"action": action})
    assert played.status_code == 200
    assert played.json()["phase"] == "setup_road"


def test_illegal_action_is_400():
    sessions.clear()
    client = TestClient(app)
    created = client.post("/api/games", json={"seed": 4}).json()
    response = client.post(f"/api/games/{created['game_id']}/actions", json={"action": {"type": "roll"}})
    assert response.status_code == 400


def test_search_is_explicit_opt_in():
    from catan_ai.search import MonteCarloBot
    client = TestClient(app)
    created = client.post('/api/games', json={'seed': 4, 'opponent': 'mc'}).json()
    session = sessions[created['game_id']]
    assert all(isinstance(bot, MonteCarloBot) for bot in session.bots.values())
    assert client.post('/api/games', json={'opponent': 'unknown'}).status_code == 422


def test_new_games_default_to_the_served_policy():
    from catan_ai.engine import BACKEND
    from catan_ai.parameter_planner import ParameterPlanner
    from catan_ai.tactical import TacticalBot
    client = TestClient(app)
    created = client.post('/api/games', json={'seed': 4}).json()
    bots = sessions[created['game_id']].bots.values()
    info = client.get('/api/bots').json()
    if BACKEND == 'python':
        # The committed checkpoint is bound to the pure-Python engine.
        assert created['opponent'] == info['default'] == 'planner_parameter' and info['served_available']
        assert all(isinstance(p, ParameterPlanner) for p in bots)
    else:
        assert created['opponent'] == info['default'] == 'tactical' and not info['served_available']
        assert all(isinstance(p, TacticalBot) for p in bots)


def test_unbound_checkpoint_falls_back_to_tactical_visibly(tmp_path, monkeypatch):
    from catan_ai.tactical import TacticalBot
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(tmp_path / 'missing.json'))
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info == {'default': 'tactical', 'served': 'planner_parameter', 'served_available': False,
                    'fallback_reason': 'Trained planner checkpoint is not bound to this server version'}
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'tactical'
    assert all(isinstance(p, TacticalBot) for p in sessions[created['game_id']].bots.values())
    assert client.post('/api/games', json={'seed': 4, 'opponent': 'planner_parameter'}).status_code == 503


import json as _json

import pytest as _pytest


@_pytest.mark.parametrize('shape', ['weights_null', 'top_level_list', 'source_missing', 'source_list', 'not_json'])
def test_structurally_invalid_checkpoint_fails_closed(tmp_path, monkeypatch, shape):
    from catan_ai import pilot_store
    from catan_ai.parameter_planner import SERVED_CHECKPOINT
    from catan_ai.tactical import TacticalBot
    monkeypatch.setenv('CATAN_PILOT_DIR', str(tmp_path / 'records'))
    client = TestClient(app)
    sessions.clear()
    # A valid game first, so restart behaviour under the broken checkpoint can be checked.
    good = client.post('/api/games', json={'seed': 4, 'opponent': 'planner_parameter'})
    if good.status_code == 200:
        game_id = good.json()['game_id']
        before = pilot_store.record_path(game_id).read_bytes()
    data = _json.loads(SERVED_CHECKPOINT.read_text())
    path = tmp_path / 'broken.json'
    if shape == 'weights_null':
        data['weights'] = None
        path.write_text(_json.dumps(data))
    elif shape == 'top_level_list':
        path.write_text(_json.dumps([data]))
    elif shape == 'source_missing':
        del data['source']
        path.write_text(_json.dumps(data))
    elif shape == 'source_list':
        data['source'] = []
        path.write_text(_json.dumps(data))
    else:
        path.write_text('{not json')
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))
    info = client.get('/api/bots')
    assert info.status_code == 200 and info.json()['served_available'] is False
    created = client.post('/api/games', json={'seed': 4})
    assert created.status_code == 200 and created.json()['opponent'] == 'tactical'
    assert all(isinstance(p, TacticalBot) for p in sessions[created.json()['game_id']].bots.values())
    records = tmp_path / 'records'
    count = len(list(records.glob('*.json')))
    assert client.post('/api/games', json={'seed': 4, 'opponent': 'planner_parameter'}).status_code == 503
    assert len(list(records.glob('*.json'))) == count
    if good.status_code == 200:
        sessions.clear()
        assert client.get(f'/api/games/{game_id}').status_code == 409
        assert pilot_store.record_path(game_id).read_bytes() == before
