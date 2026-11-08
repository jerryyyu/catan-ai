"""NumPy serving of the arch-bakeoff tiles transformer and the ``planner_pv2_tiles`` bot."""
import json
from pathlib import Path
import shutil
import sys
import time

import pytest

np = pytest.importorskip('numpy')

from catan_ai import tiles_value as tv  # noqa: E402
from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.experiments import source_files  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from catan_ai.parameter_planner import DEFAULT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / 'checkpoints' / 'value_tiles_v1.pt'
sys.path.insert(0, str(ROOT / 'scripts'))
import arch_bakeoff as ab  # noqa: E402

needs_weights = pytest.mark.skipif(not tv.TILES_V1.with_suffix('.npz').exists(),
                                   reason='tiles weights are a local artifact (not committed)')


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def test_layout_constants_match_arch_bakeoff():
    for name in ('N_TILES', 'N_VERTICES', 'N_EDGES', 'N_PLAYERS', 'TILE_W', 'VERTEX_W', 'EDGE_W', 'PLAYER_W',
                 'GLOBAL_W', 'V0', 'E0', 'P0', 'G0', 'WIDTH', 'TOKEN_W'):
        assert getattr(tv, name) == getattr(ab, name), name
    assert (tv.D, tv.HEADS, tv.LAYERS) == (ab.TILES_D, ab.TILES_HEADS, ab.TILES_LAYERS)


def test_tokenizer_equals_arch_bakeoff_on_random_inputs():
    board = ab.layout()
    tidx = np.asarray(tv.tile_index(board))
    assert np.array_equal(tidx, ab.tile_index(board))
    x = np.random.default_rng(3).standard_normal((64, tv.WIDTH)).astype(np.float32)
    tile, player, glob = tv.tokenize(np, x, tidx)
    assert tile.shape == (64, 19, 116) and player.shape == (64, 4, 27) and glob.shape == (64, 20)
    torch = pytest.importorskip('torch')
    xt = torch.from_numpy(x)
    ref = (xt[:, torch.from_numpy(ab.tile_index(board))], xt[:, ab.P0:ab.G0].reshape(64, ab.N_PLAYERS, ab.PLAYER_W),
           xt[:, ab.G0:])
    for got, want in zip((tile, player, glob), ref):
        assert np.array_equal(got.astype(np.float32), want.numpy())


@needs_weights
def test_numpy_forward_matches_torch_on_random_standardized_rows():
    torch = pytest.importorskip('torch')
    if not PT.exists():
        pytest.skip('torch checkpoint is a local artifact')
    ckpt = torch.load(PT, map_location='cpu', weights_only=False)
    net = ab.make_net(torch, 'tiles', ab.layout())
    net.load_state_dict(ckpt['state'])
    net.eval()
    z = np.random.default_rng(11).standard_normal((256, tv.WIDTH)).astype(np.float32)
    with torch.no_grad():
        want = net(torch.from_numpy(z)).double().numpy()
    served = tv.TilesValueNet()
    got = served.forward_standardized(z.astype(np.float64))
    diff = float(np.max(np.abs(got - want)))
    print(f'parity max abs logit diff (256 random standardized rows): {diff:.3g}')
    assert diff < 1e-4
    # the raw-row path standardizes exactly like arch_bakeoff.to_batch
    rows = served.mean + served.sd * z[:16]
    with torch.no_grad():
        want = net((torch.from_numpy(rows.astype(np.float32)) - ckpt['mean']) / ckpt['sd']).double().numpy()
    assert np.max(np.abs(served.logits(rows) - want)) < 1e-4


@needs_weights
def test_loader_refuses_a_tampered_npz(tmp_path):
    stem = tmp_path / 'value_tiles_v1'
    shutil.copy(tv.TILES_V1.with_suffix('.meta.json'), f'{stem}.meta.json')
    shutil.copy(tv.TILES_V1.with_suffix('.npz'), f'{stem}.npz')
    tv.TilesValueNet(stem)  # an untouched copy loads
    raw = bytearray(Path(f'{stem}.npz').read_bytes())
    raw[len(raw) // 2] ^= 1
    Path(f'{stem}.npz').write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='do not match their metadata'):
        tv.TilesValueNet(stem)


@needs_weights
def test_pv2_tiles_plays_a_full_seeded_game_against_tactical_with_legal_actions():
    game = Game.new(list('abcd'), 2690007)
    bots = [make_policy('planner_pv2_tiles', 1)] + [make_policy('tactical', i) for i in (2, 3, 4)]
    times, sampled, offers = [], 0, 0
    while game.winner is None and game.turn < 1000:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        started = time.perf_counter()
        action = bots[seat].choose(view, seat)
        if seat == 0:
            times.append(time.perf_counter() - started)
            if action['type'] == 'offer_trade':
                offers += 1  # offers are validated by the engine, not enumerated in view.actions
            else:
                assert action in view.actions
                if view.phase == 'actions' and sampled < 20:
                    sampled += 1
        game.apply(seat, action)
    assert game.winner is not None
    assert sampled == 20
    p50, p95 = np.percentile(times, [50, 95])
    print(f'pv2_tiles: winner {game.winner} turn {game.turn}; {len(times)} decisions, {offers} offers; '
          f'decision p50 {p50:.4f}s p95 {p95:.4f}s max {max(times):.4f}s')
    assert p95 < .5


@needs_weights
def test_pv2_tiles_scores_with_the_tiles_net_and_keeps_the_base_integration():
    from catan_ai.pv_search import PVSearchTilesPlanner, PVSearchV2BFixPlanner
    bot = make_policy('planner_pv2_tiles', 1)
    assert isinstance(bot, PVSearchTilesPlanner) and isinstance(bot.net, tv.TilesValueNet)
    assert PVSearchTilesPlanner.choose is PVSearchV2BFixPlanner.choose
    x = np.random.default_rng(5).standard_normal((3, tv.WIDTH))
    assert bot._logits(list(x)) == [float(v) for v in bot.net.logits(x)] and bot._logits([]) == []
