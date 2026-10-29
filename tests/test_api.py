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
