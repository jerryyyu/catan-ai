"""Serving chain for the 2026-11-11 default:
planner_pv2_tiles_va_t9 -> planner_pv2_tiles -> planner_offer_accept -> Tactical.

Both value-weight files are local artifacts (never committed), so CI has none. The
positive paths use synthetic exports with the real layout (random weights, their own
parity probe), written in the format ``tiles_value.TilesValueNet`` loads; the T9
export carries the gate-scored .pt SHA256 from ``value_arms_serving.ARMS``. Every
test points both stems at a temporary directory so local weights never leak in; the
real-weights tests at the end restore the real stems when the files are present.
"""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catan_ai import api, tiles_value, value_arms_serving
from catan_ai.api import app, sessions
from catan_ai.engine import BACKEND

np = pytest.importorskip('numpy')

python_only = pytest.mark.skipif(BACKEND != 'python', reason='the fitted checkpoint is bound to the pure-Python engine')
REAL_STEM = tiles_value.TILES_V1
REAL_CHECKPOINTS = value_arms_serving.CHECKPOINTS
REAL_T9_STEM = REAL_CHECKPOINTS / 'value_tiles_va_t9'
T9_REASON = 'T9 value weights missing or unverified'
TILES_REASON = 'Tiles value weights missing or unverified'
UNBOUND_REASON = 'Trained planner checkpoint is not bound to this server version'


def write_synthetic_tiles(stem: Path, seed: int = 0, meta_from: Path = REAL_STEM) -> str:
    """Write <stem>.npz/.meta.json with the real array shapes and a matching parity probe."""
    meta = json.loads(Path(f'{meta_from}.meta.json').read_text())
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
    (tmp_path / 'ck').mkdir()
    monkeypatch.setattr(tiles_value, 'TILES_V1', tmp_path / 'absent')
    monkeypatch.setattr(value_arms_serving, 'CHECKPOINTS', tmp_path / 'ck')
    api._value_weights_cache.clear()
    sessions.clear()
    yield
    api._value_weights_cache.clear()
    sessions.clear()


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Synthetic planner_pv2_tiles weights (the T9 stem stays absent)."""
    stem = tmp_path / 'value_tiles_synthetic'
    sha = write_synthetic_tiles(stem)
    monkeypatch.setattr(tiles_value, 'TILES_V1', stem)
    return stem, sha


@pytest.fixture
def t9(tmp_path):
    """Synthetic T9 weights at the stem value_arms_serving loads."""
    stem = tmp_path / 'ck' / 'value_tiles_va_t9'
    sha = write_synthetic_tiles(stem, seed=7, meta_from=REAL_T9_STEM)
    return stem, sha


def bots_info():
    return TestClient(app).get('/api/bots').json()


@python_only
def test_t9_served_when_its_weights_verify(synthetic, t9):
    stem, sha = t9
    assert api.served_default() == ('planner_pv2_tiles_va_t9', None)
    client = TestClient(app)
    assert client.get('/api/bots').json() == {
        'default': 'planner_pv2_tiles_va_t9', 'served': 'planner_pv2_tiles_va_t9', 'served_available': True,
        'fallback_reason': None, 'fallback_detail': None, 'checkpoint_available': True, 'tiles_available': True}
    created = client.post('/api/games', json={'seed': 4})
    assert created.status_code == 200 and created.json()['opponent'] == 'planner_pv2_tiles_va_t9'
    session = sessions[created.json()['game_id']]
    assert all(isinstance(bot, value_arms_serving.ValueArmPlanner) and bot.ARM == 't9'
               for bot in session.bots.values())
    assert session.metadata['value_sha256'] == sha and session.metadata['model_sha256']


@python_only
def test_t9_served_without_tiles_weights(t9):
    info = bots_info()
    assert info['default'] == 'planner_pv2_tiles_va_t9' and info['served_available'] is True
    assert info['tiles_available'] is False


@python_only
def test_planner_pv2_tiles_stays_selectable(synthetic, t9):
    from catan_ai.pv_search import PVSearchTilesPlanner
    _, sha = synthetic
    created = TestClient(app).post('/api/games', json={'seed': 4, 'opponent': 'planner_pv2_tiles'})
    assert created.status_code == 200 and created.json()['opponent'] == 'planner_pv2_tiles'
    session = sessions[created.json()['game_id']]
    assert all(type(bot) is PVSearchTilesPlanner for bot in session.bots.values())
    assert session.metadata['value_sha256'] == sha


@python_only
def test_weights_verified_once_per_stem_not_per_request(synthetic, t9, monkeypatch):
    calls = []
    real = tiles_value.TilesValueNet

    def counting(stem):
        calls.append(Path(stem).name)
        return real(stem)
    monkeypatch.setattr(tiles_value, 'TilesValueNet', counting)
    client = TestClient(app)
    for _ in range(3):
        assert client.get('/api/bots').json()['default'] == 'planner_pv2_tiles_va_t9'
    assert sorted(calls) == ['value_tiles_synthetic', 'value_tiles_va_t9']


@python_only
def test_missing_t9_weights_fall_back_to_tiles(synthetic, tmp_path):
    from catan_ai.pv_search import PVSearchTilesPlanner
    _, sha = synthetic
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info == {
        'default': 'planner_pv2_tiles', 'served': 'planner_pv2_tiles_va_t9', 'served_available': False,
        'fallback_reason': T9_REASON,
        'fallback_detail': "FileNotFoundError: [Errno 2] No such file or directory: 'value_tiles_va_t9.meta.json'",
        'checkpoint_available': True, 'tiles_available': True}
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'planner_pv2_tiles'
    session = sessions[created['game_id']]
    assert all(type(bot) is PVSearchTilesPlanner for bot in session.bots.values())
    assert session.metadata['value_sha256'] == sha
    records = tmp_path / 'records'
    count = len(list(records.glob('*.json')))
    assert client.post('/api/games', json={'seed': 4, 'opponent': 'planner_pv2_tiles_va_t9'}).status_code == 503
    assert len(list(records.glob('*.json'))) == count


@python_only
def test_t9_weights_with_wrong_sha_fall_back_to_tiles(synthetic, t9):
    stem, _ = t9
    raw = bytearray(Path(f'{stem}.npz').read_bytes())
    raw[-1] ^= 1
    Path(f'{stem}.npz').write_bytes(bytes(raw))
    info = bots_info()
    assert info['default'] == 'planner_pv2_tiles' and info['fallback_reason'] == T9_REASON
    assert info['fallback_detail'] == 'ValueError: tiles value weights do not match their metadata'


@python_only
def test_t9_weights_from_another_checkpoint_fall_back(synthetic, tmp_path):
    # A self-consistent export of some other .pt (here: the tiles v1 metadata) is not T9.
    write_synthetic_tiles(tmp_path / 'ck' / 'value_tiles_va_t9', seed=7, meta_from=REAL_STEM)
    info = bots_info()
    assert info['default'] == 'planner_pv2_tiles' and info['fallback_reason'] == T9_REASON
    assert info['fallback_detail'] == 'ValueError: T9 value weights were not exported from the gate-scored checkpoint'


@python_only
def test_both_weights_missing_fall_back_to_offer_accept(tmp_path):
    from catan_ai.unified_planner import AcceptanceOfferPlanner
    Path(f'{tmp_path / "absent"}.meta.json').write_text(Path(f'{REAL_STEM}.meta.json').read_text())
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info == {
        'default': 'planner_offer_accept', 'served': 'planner_pv2_tiles_va_t9', 'served_available': False,
        'fallback_reason': TILES_REASON,
        'fallback_detail': "FileNotFoundError: [Errno 2] No such file or directory: 'absent.npz'",
        'checkpoint_available': True, 'tiles_available': False}
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'planner_offer_accept'
    assert all(type(bot) is AcceptanceOfferPlanner for bot in sessions[created['game_id']].bots.values())
    records = tmp_path / 'records'
    count = len(list(records.glob('*.json')))
    for name in ('planner_pv2_tiles_va_t9', 'planner_pv2_tiles'):
        assert client.post('/api/games', json={'seed': 4, 'opponent': name}).status_code == 503
    assert len(list(records.glob('*.json'))) == count


@python_only
def test_tiles_weights_with_wrong_sha_fall_back_to_offer_accept(synthetic):
    stem, _ = synthetic
    raw = bytearray(Path(f'{stem}.npz').read_bytes())
    raw[-1] ^= 1
    Path(f'{stem}.npz').write_bytes(bytes(raw))
    info = bots_info()
    assert info['default'] == 'planner_offer_accept' and info['fallback_reason'] == TILES_REASON
    assert info['fallback_detail'] == 'ValueError: tiles value weights do not match their metadata'


@python_only
def test_replacing_weights_is_rechecked(synthetic, t9):
    t9_stem, _ = t9
    tiles_stem, _ = synthetic
    client = TestClient(app)
    assert client.get('/api/bots').json()['default'] == 'planner_pv2_tiles_va_t9'
    Path(f'{t9_stem}.npz').unlink()
    assert client.get('/api/bots').json()['default'] == 'planner_pv2_tiles'
    Path(f'{tiles_stem}.npz').unlink()
    assert client.get('/api/bots').json()['default'] == 'planner_offer_accept'


@python_only
def test_numpy_absent_falls_back(synthetic, t9, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'numpy', None)
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info['default'] == 'planner_offer_accept' and info['served_available'] is False
    assert info['fallback_reason'] == TILES_REASON
    assert info['fallback_detail'].startswith('ImportError') or info['fallback_detail'].startswith('ModuleNotFound')
    assert client.post('/api/games', json={'seed': 4}).json()['opponent'] == 'planner_offer_accept'
    for name in ('planner_pv2_tiles_va_t9', 'planner_pv2_tiles'):
        assert client.post('/api/games', json={'seed': 4, 'opponent': name}).status_code == 503


def test_unbound_checkpoint_falls_back_to_tactical(synthetic, t9, tmp_path, monkeypatch):
    from catan_ai.tactical import TacticalBot
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(tmp_path / 'missing.json'))
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert {k: v for k, v in info.items() if k != 'fallback_detail'} == {
        'default': 'tactical', 'served': 'planner_pv2_tiles_va_t9', 'served_available': False,
        'fallback_reason': UNBOUND_REASON, 'checkpoint_available': False, 'tiles_available': False}
    assert info['fallback_detail'] == "FileNotFoundError: [Errno 2] No such file or directory: 'missing.json'"
    assert str(tmp_path) not in json.dumps(info)
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'tactical'
    assert all(isinstance(bot, TacticalBot) for bot in sessions[created['game_id']].bots.values())
    for name in ('planner_pv2_tiles_va_t9', 'planner_pv2_tiles'):
        assert client.post('/api/games', json={'seed': 4, 'opponent': name}).status_code == 503


@python_only
def test_restart_recovery_checks_the_t9_weights(synthetic, t9):
    stem, sha = t9
    client = TestClient(app)
    created = client.post('/api/games', json={'seed': 4}).json()
    game_id = created['game_id']
    assert created['opponent'] == 'planner_pv2_tiles_va_t9'
    assert api.pilot_store.load(game_id)['value_sha256'] == sha
    sessions.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 200
    sessions.clear()
    write_synthetic_tiles(stem, seed=8, meta_from=REAL_T9_STEM)  # different weights, self-consistent metadata
    api._value_weights_cache.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 409


@python_only
def test_restart_recovery_checks_the_tiles_weights(synthetic):
    stem, _ = synthetic
    client = TestClient(app)
    created = client.post('/api/games', json={'seed': 4}).json()
    game_id = created['game_id']
    assert created['opponent'] == 'planner_pv2_tiles'
    sessions.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 200
    sessions.clear()
    write_synthetic_tiles(stem, seed=1)
    api._value_weights_cache.clear()
    assert client.get(f'/api/games/{game_id}').status_code == 409


@python_only
@pytest.mark.skipif(not Path(f'{REAL_STEM}.npz').exists(), reason='local tiles weights absent (never committed)')
def test_real_local_tiles_weights_are_served(monkeypatch):
    monkeypatch.setattr(tiles_value, 'TILES_V1', REAL_STEM)
    meta = json.loads(Path(f'{REAL_STEM}.meta.json').read_text())
    assert meta['npz_sha256'] == '27dd7818cdc29082dd5c712c0f1fdf708ff568cdc98fb4b898f01e3f60820697'
    client = TestClient(app)
    info = client.get('/api/bots').json()  # the T9 stem is still the empty temporary one
    assert info['default'] == 'planner_pv2_tiles' and info['tiles_available'] is True
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'planner_pv2_tiles'
    assert sessions[created['game_id']].metadata['value_sha256'] == meta['npz_sha256']


@python_only
@pytest.mark.skipif(not Path(f'{REAL_T9_STEM}.npz').exists(), reason='local T9 weights absent (never committed)')
def test_real_local_t9_weights_are_served(monkeypatch):
    monkeypatch.setattr(value_arms_serving, 'CHECKPOINTS', REAL_CHECKPOINTS)
    meta = json.loads(Path(f'{REAL_T9_STEM}.meta.json').read_text())
    assert meta['npz_sha256'] == 'a20be3e112c900653f73ec71aeda384f5ba6bcf36303fc189dcc7832fe8891a5'
    client = TestClient(app)
    info = client.get('/api/bots').json()
    assert info['default'] == 'planner_pv2_tiles_va_t9' and info['served_available'] is True
    created = client.post('/api/games', json={'seed': 4}).json()
    assert created['opponent'] == 'planner_pv2_tiles_va_t9'
    assert sessions[created['game_id']].metadata['value_sha256'] == meta['npz_sha256']


def test_fallback_detail_hides_server_paths():
    assert api.public_detail("OSError: cannot read '/srv/catan/checkpoints/value_tiles_v1.npz' now") == \
        "OSError: cannot read 'value_tiles_v1.npz' now"
    assert api.public_detail('ValueError: tiles value weights do not match their metadata') == \
        'ValueError: tiles value weights do not match their metadata'
    assert api.public_detail(None) is None
