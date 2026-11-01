from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/audit_solo_pilot.py'))['audit']


def record_for(human, opponent='tactical', complete=False):
    seed = 2600300
    game = Game.new(['PRIVATE NAME'] * 4, seed)
    policies = [make_policy('heuristic' if s == human else opponent, seed * 31 + s) for s in range(4)]
    trace = []
    for _ in range(10000 if complete else 20):
        if game.winner is not None:
            break
        seat = game.current
        action = policies[seat].choose(Observation.for_policy(game, seat, policies[seat]), seat)
        game.apply(seat, action)
        trace.append({'seat': seat, 'action': action, 'seconds': .01,
                      'actor': 'human' if seat == human else 'bot'})
    record = {'schema': 2, 'source': source_files(), 'human_seat': human, 'seed': seed,
              'names': ['PRIVATE NAME'] * 4, 'opponent': opponent, 'trace': trace,
              'winner': game.winner, 'turn': game.turn}
    if opponent in ('planner_parameter', 'planner_devfix', 'planner_unified'):
        record['model_sha256'] = policies[(human + 1) % 4].model_hash
    return record


@pytest.mark.parametrize('seat', range(4))
def test_partial_solo_audit_is_read_only_and_does_not_certify_humans(seat):
    record = record_for(seat)
    before = deepcopy(record)
    result = audit(record)
    assert result['replay_valid'] and not result['completed']
    assert not result['human_participation_verified'] and not result['experience_verified']
    assert result['human_won'] is None and 'PRIVATE NAME' not in str(result)
    assert record == before


@pytest.mark.parametrize('defect', ['source', 'seat', 'actor', 'timing', 'winner', 'turn', 'action'])
def test_solo_audit_rejects_corruption(defect):
    record = record_for(3)
    if defect == 'source':
        record['source'] = {}
    elif defect == 'seat':
        record['human_seat'] = True
    elif defect == 'actor':
        record['trace'][0]['actor'] = 'human'
    elif defect == 'timing':
        record['trace'][0]['seconds'] = float('nan')
    elif defect == 'winner':
        record['winner'] = 1
    elif defect == 'turn':
        record['turn'] += 1
    else:
        record['trace'][0]['action'] = {'type': 'roll'}
    with pytest.raises(ValueError):
        audit(record)


def test_complete_synthetic_game_has_outcome_but_is_not_human_evidence():
    record = record_for(0, complete=True)
    assert record['winner'] is not None
    result = audit(record)
    assert result['completed'] and result['human_won'] == (record['winner'] == 0)
    assert result['recorded_human_actions'] > 0 and result['recorded_bot_actions'] > 0
    assert not result['human_participation_verified']


@pytest.mark.parametrize('opponent', ['planner_parameter', 'planner_devfix', 'planner_unified'])
def test_fitted_audit_rejects_missing_or_changed_checkpoint_identity(tmp_path, monkeypatch, opponent):
    import json
    from catan_ai.parameter_planner import DEFAULT
    path = tmp_path / 'synthetic-model.json'
    data = {'schema': 1, 'source': source_files(), 'weights': DEFAULT}
    path.write_text(json.dumps(data))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))
    record = record_for(1, opponent=opponent)
    assert audit(record)['replay_valid']
    missing = deepcopy(record)
    missing.pop('model_sha256')
    with pytest.raises(ValueError, match='checkpoint'):
        audit(missing)
    data['metadata_only_change'] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='checkpoint'):
        audit(record)


@pytest.mark.parametrize('opponent', ['tactical', 'planner_parameter', 'planner_devfix'])
def test_actual_solo_api_record_audits_without_rewriting_file(tmp_path, monkeypatch, opponent):
    from fastapi.testclient import TestClient
    from catan_ai import api, pilot_store
    monkeypatch.setenv('CATAN_PILOT_DIR', str(tmp_path))
    if opponent != 'tactical':
        import json
        from catan_ai.parameter_planner import DEFAULT
        model = tmp_path / 'synthetic-model.json'
        model.write_text(json.dumps({'schema': 1, 'source': source_files(), 'weights': DEFAULT}))
        monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(model))
    client = TestClient(api.app)
    created = client.post('/api/games', json={'seed': 2600301, 'human_seat': 1, 'opponent': opponent})
    assert created.status_code == 200
    game_id = created.json()['game_id']
    try:
        path = pilot_store.record_path(game_id)
        before = path.read_bytes()
        result = audit(pilot_store.load(game_id))
        assert result['replay_valid'] and result['recorded_bot_actions'] == 2
        assert path.read_bytes() == before
    finally:
        api.sessions.pop(game_id, None)
