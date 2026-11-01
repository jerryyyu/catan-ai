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
