"""Serving chain for the 2026-11-08 default: planner_pv2_tiles -> planner_offer_accept -> Tactical.

The tiles weights are a local artifact (never committed), so CI has none. The positive
path uses a synthetic export with the real layout (random weights, its own parity probe),
written in the format ``tiles_value.TilesValueNet`` loads; one test also runs against the
real local weights when they are present.
"""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catan_ai import api, tiles_value
from catan_ai.api import app, sessions
from catan_ai.engine import BACKEND

np = pytest.importorskip('numpy')

python_only = pytest.mark.skipif(BACKEND != 'python', reason='the fitted checkpoint is bound to the pure-Python engine')
REAL_STEM = tiles_value.TILES_V1


def write_synthetic_tiles(stem: Path, seed: int = 0) -> str:
    """Write <stem>.npz/.meta.json with the real array shapes and a matching parity probe."""
    meta = json.loads(Path(f'{REAL_STEM}.meta.json').read_text())
    rng = np.random.default_rng(seed)
    arrays = {}
    for name, shape in meta['arrays'].items():
        if name.startswith('w:'):
            arrays[name] = rng.normal(0., .05, shape)
    arrays['mean'] = rng.normal(0., .1, meta['arrays']['mean'])
    arrays['sd'] = rng.uniform(.5, 1.5, meta['arrays']['sd'])
    arrays['tidx'] = np.asarray(tiles_value.tile_index(), dtype=np.int64)
    arrays['types'] = np.asarray([0] * tiles_value.N_TILES + [1] * tiles_value.N_PLAYERS + [2])
    arrays['parity_x'] = rng.normal(0., 1., (4, tiles_value.WIDTH))
    net = object.__new__(tiles_value.TilesValueNet)
    net.np = np
    net.w = {k[2:]: v for k, v in arrays.items() if k.startswith('w:')}
    net.mean, net.sd, net.tidx = arrays['mean'], arrays['sd'], arrays['tidx']
    net.embed = net.w['pos'] + net.w['kind'][arrays['types']]
    arrays['parity_logit'] = net.logits(arrays['parity_x'])
    np.savez(f'{stem}.npz', **arrays)
    sha = hashlib.sha256(Path(f'{stem}.npz').read_bytes()).hexdigest()
    Path(f'{stem}.meta.json').write_text(json.dumps(dict(meta, npz_sha256=sha, source_run='synthetic test fixture')))
    return sha


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv('CATAN_PILOT_DIR', str(tmp_path / 'records'))
    api._value_weights_cache.clear()
    sessions.clear()
    yield
    api._value_weights_cache.clear()
    sessions.clear()


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    stem = tmp_path / 'value_tiles_synthetic'
    sha = write_synthetic_tiles(stem)
    monkeypatch.setattr(tiles_value, 'TILES_V1', stem)
    return stem, sha


@python_only
def test_served_when_weights_verify(synthetic):
    from catan_ai.pv_search import PVSearchTilesPlanner
    stem, sha = synthetic
    assert api.served_default() == ('planner_pv2_tiles', None)
    client = TestClient(app)
    assert client.get('/api/bots').json() == {
        'default': 'planner_pv2_tiles', 'served': 'planner_pv2_tiles', 'served_available': True,
        'fallback_reason': None, 'fallback_detail': None, 'checkpoint_available': True}
    created = client.post('/api/games', json={'seed': 4})
    assert created.status_code == 200 and created.json()['opponent'] == 'planner_pv2_tiles'
    session = sessions[created.json()['game_id']]
    assert all(isinstance(bot, PVSearchTilesPlanner) for bot in session.bots.values())
    assert session.metadata['value_sha256'] == sha and session.metadata['model_sha256']


@python_only
def test_weights_verified_once_not_per_request(synthetic, monkeypatch):
    calls = []
    real = tiles_value.TilesValueNet

    def counting(stem):
        calls.append(stem)
        return real(stem)
    monkeypatch.setattr(tiles_value, 'TilesValueNet', counting)
    client = TestClient(app)
    for _ in range(3):
        assert client.get('/api/bots').json()['default'] == 'planner_pv2_tiles'
    assert len(calls) == 1


@python_only
def test_missing_weights_fall_back_to_offer_accept(tmp_path, monkeypatch):
    from catan_ai.unified_planner import AcceptanceOfferPlanner
    stem = tmp_path / 'absent'
    Path(f'{stem}.meta.json').write_text(Path(f'{REAL_STEM}.meta.json').read_text())
    monkeypatch.setattr(tiles_value, 'TILES_V1', stem)
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info['default'] == 'planner_offer_accept' and info['served'] == 'planner_pv2_tiles'
    assert info['served_available'] is False and info['checkpoint_available'] is True
    assert info['fallback_reason'] == 'Tiles value weights missing or unverified'
    assert info['fallback_detail'].startswith('FileNotFoundError')
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'planner_offer_accept'
    assert all(type(bot) is AcceptanceOfferPlanner for bot in sessions[created['game_id']].bots.values())
    records = tmp_path / 'records'
    count = len(list(records.glob('*.json')))
    assert client.post('/api/games', json={'seed': 4, 'opponent': 'planner_pv2_tiles'}).status_code == 503
    assert len(list(records.glob('*.json'))) == count


@python_only
def test_weights_with_wrong_sha_fall_back(synthetic):
    stem, _ = synthetic
    raw = bytearray(Path(f'{stem}.npz').read_bytes())
    raw[-1] ^= 1
    Path(f'{stem}.npz').write_bytes(bytes(raw))
    info = TestClient(app).get('/api/bots').json()
    assert info['default'] == 'planner_offer_accept'
    assert info['fallback_reason'] == 'Tiles value weights missing or unverified'
    assert info['fallback_detail'] == 'ValueError: tiles value weights do not match their metadata'


@python_only
def test_replacing_weights_is_rechecked(synthetic):
    stem, _ = synthetic
    client = TestClient(app)
    assert client.get('/api/bots').json()['default'] == 'planner_pv2_tiles'
    Path(f'{stem}.npz').unlink()
    assert client.get('/api/bots').json()['default'] == 'planner_offer_accept'


@python_only
def test_numpy_absent_falls_back(synthetic, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'numpy', None)
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info['default'] == 'planner_offer_accept' and info['served_available'] is False
    assert info['fallback_detail'].startswith('ImportError') or info['fallback_detail'].startswith('ModuleNotFound')
    assert client.post('/api/games', json={'seed': 4}).json()['opponent'] == 'planner_offer_accept'
    assert client.post('/api/games', json={'seed': 4, 'opponent': 'planner_pv2_tiles'}).status_code == 503


def test_unbound_checkpoint_falls_back_to_tactical(synthetic, tmp_path, monkeypatch):
    from catan_ai.tactical import TacticalBot
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(tmp_path / 'missing.json'))
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert {k: v for k, v in info.items() if k != 'fallback_detail'} == {
        'default': 'tactical', 'served': 'planner_pv2_tiles', 'served_available': False,
        'fallback_reason': 'Trained planner checkpoint is not bound to this server version',
        'checkpoint_available': False}
    assert info['fallback_detail'].startswith('FileNotFoundError')
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'tactical'
    assert all(isinstance(bot, TacticalBot) for bot in sessions[created['game_id']].bots.values())
    assert client.post('/api/games', json={'seed': 4, 'opponent': 'planner_pv2_tiles'}).status_code == 503


@python_only
def test_restart_recovery_checks_the_value_weights(synthetic):
    stem, _ = synthetic
    client = TestClient(app)
    created = client.post('/api/games', json={'seed': 4}).json()
    game_id = created['game_id']
    assert created['opponent'] == 'planner_pv2_tiles'
    sessions.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 200
    sessions.clear()
    write_synthetic_tiles(stem, seed=1)  # different weights, self-consistent metadata
    api._value_weights_cache.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 409


@python_only
@pytest.mark.skipif(not Path(f'{REAL_STEM}.npz').exists(), reason='local tiles weights absent (never committed)')
def test_real_local_weights_are_served():
    meta = json.loads(Path(f'{REAL_STEM}.meta.json').read_text())
    assert meta['npz_sha256'] == '27dd7818cdc29082dd5c712c0f1fdf708ff568cdc98fb4b898f01e3f60820697'
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info['default'] == 'planner_pv2_tiles' and info['served_available'] is True
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'planner_pv2_tiles'
    assert sessions[created['game_id']].metadata['value_sha256'] == meta['npz_sha256']
