"""Architecture bake-off script: tokenization, board planes, arms and a synthetic smoke run."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import pytest

np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import arch_bakeoff as ab  # noqa: E402
from train_value import auc  # noqa: E402
from train_value_v2 import DATA_KEYS, expected_shards, shards, symmetry_index  # noqa: E402

CONFIG = ROOT / 'experiments' / 'arch-bakeoff.json'


def make_fixture(tmp, train_games=(14, 6), test_games=10, rows_per_game=16, seed=0):
    """Synthetic shards + game records consistent with the declared split and encoder layout,
    written as ``reencode_v2.py`` does (full shard inventory, NPZ SHA-256 in each sidecar) so
    the verified loader ``train_value_v2.shards`` accepts them. Train games:
    ``train_games[0]`` before the validation cut, ``train_games[1]`` after it; every other
    seed of each block is an empty shard slot with a placeholder source hash."""
    rng = np.random.default_rng(seed)
    data, runs = tmp / 'shards', tmp / 'runs'
    data.mkdir()
    ref = json.loads((ROOT / 'experiments' / 'value-v2-100k-aug-b.json').read_text())
    done = {k: ref[k] for k in DATA_KEYS}
    (data / 'DONE.json').write_text(json.dumps({'config': done}))
    (lo, n), (tlo, _) = ref['split']['train'], ref['split']['test']
    cut = lo + n - 5000
    splits = {'train': [lo + k for k in range(train_games[0])] + [cut + k for k in range(train_games[1])],
              'test': [tlo + k for k in range(test_games)]}
    for split, seeds in splits.items():
        xs, ys, gs, hashes = [], [], [], {}
        for i, s in enumerate(seeds):
            truncated = i == 1
            winner = None if truncated else int(rng.integers(4))
            turns = 3 * rows_per_game + 5
            root = runs / ('value100k-v1' if i % 2 else 'value100k-v1b') / 'games'
            root.mkdir(parents=True, exist_ok=True)
            raw = json.dumps({'seed': s, 'winner': winner, 'turns': turns, 'value_rows': []}).encode()
            (root / f'{s}.json').write_bytes(raw)
            hashes[str(s)] = hashlib.sha256(raw).hexdigest()
            for r in range(rows_per_game):
                seat = r % 4
                x = rng.normal(size=ab.WIDTH)
                x[ab.SEAT_COLS] = 0
                x[ab.SEAT_COLS.start + seat] = 1
                x[ab.TURN_COL] = (3 * r + 1) / 400
                won = winner == seat
                x[0] += 1.5 * won  # a little signal
                xs.append(x)
                ys.append(int(won))
                gs.append(s)
        x, y, g = np.asarray(xs, dtype=np.float16), np.asarray(ys, dtype=np.uint8), np.asarray(gs, dtype=np.int32)
        for stem, (first, last) in expected_shards(done, split).items():
            keep = (g >= first) & (g <= last)
            npz = data / f'{stem}.npz'
            with npz.open('wb') as stream:
                np.savez(stream, x=x[keep].reshape(-1, ab.WIDTH), y=y[keep], game=g[keep],
                         leaf=np.zeros(int(keep.sum()), dtype=np.float32),
                         shared=np.zeros(int(keep.sum()), dtype=np.float32))
            meta = {'split': split, 'seeds': [first, last], 'rows': int(keep.sum()),
                    'npz_sha256': hashlib.sha256(npz.read_bytes()).hexdigest(),
                    'source_sha256': {str(s): hashes.get(str(s), 'ab' * 32) for s in range(first, last + 1)}}
            if keep.any():
                meta['row_games'] = [int(g[keep].min()), int(g[keep].max())]
            (data / f'{stem}.json').write_text(json.dumps(meta))
    return data, runs


@pytest.fixture(scope='module')
def board():
    return ab.layout()


def test_layout_offsets_match_encoder():
    from catan_ai.encoding import size
    assert size() == ab.WIDTH == 1324
    assert (ab.V0, ab.E0, ab.P0, ab.G0) == (152, 908, 1196, 1304)
    assert ab.TURN_COL == ab.WIDTH - 10 and ab.SEAT_COLS == slice(ab.WIDTH - 9, ab.WIDTH - 5)


def test_tile_tokens_hold_exactly_their_corners_and_sides(board):
    index = ab.tile_index(board)
    assert index.shape == (19, ab.TOKEN_W)
    edges = {frozenset(e): i for i, e in enumerate(board.edges)}
    for t, tile in enumerate(board.tiles):
        row = index[t]
        assert list(row[:8]) == list(range(8 * t, 8 * t + 8))
        corners = []
        for k in range(6):
            block = row[8 + 14 * k: 8 + 14 * (k + 1)]
            v = (block[0] - ab.V0) // 14
            assert list(block) == list(range(ab.V0 + 14 * v, ab.V0 + 14 * v + 14))
            corners.append(int(v))
        assert sorted(corners) == sorted(tile.vertices)
        sides = []
        for k in range(6):
            block = row[8 + 84 + 4 * k: 8 + 84 + 4 * (k + 1)]
            e = (block[0] - ab.E0) // 4
            assert list(block) == list(range(ab.E0 + 4 * e, ab.E0 + 4 * e + 4))
            sides.append(int(e))
            assert e == edges[frozenset((corners[k], corners[(k + 1) % 6]))]
        direct = {i for i, (a, b) in enumerate(board.edges) if a in tile.vertices and b in tile.vertices}
        assert set(sides) == direct and len(direct) == 6
        cx = sum(board.vertices[v][0] for v in tile.vertices) / 6
        cy = sum(board.vertices[v][1] for v in tile.vertices) / 6
        angles = [math.atan2(board.vertices[v][1] - cy, board.vertices[v][0] - cx) % (2 * math.pi) for v in corners]
        assert all(b > a for a, b in zip(angles, angles[1:]))  # counterclockwise


def test_grid_places_every_board_feature_once_without_collisions(board):
    src, (h, w) = ab.grid_index(board)
    assert src.shape == (ab.CHANNELS * h * w,)
    placed = np.sort(src[src < ab.WIDTH])
    assert np.array_equal(placed, np.arange(ab.P0))
    planes = src.reshape(ab.CHANNELS, h, w)
    groups = ((0, 8, 0, 8), (8, 22, ab.V0, 14), (22, 26, ab.E0, 4))
    occupied = 0
    for r in range(h):
        for c in range(w):
            cell = planes[:, r, c]
            used = [g for g in groups if np.any(cell[g[0]:g[1]] < ab.WIDTH)]
            assert len(used) <= 1
            for a, b, first, width in used:
                block = cell[a:b]
                assert np.all(block < ab.WIDTH) and np.all(cell[:a] == ab.WIDTH) and np.all(cell[b:] == ab.WIDTH)
                assert (block[0] - first) % width == 0 and list(block) == list(range(block[0], block[0] + width))
                occupied += 1
    assert occupied == 19 + 54 + 72
    # the CNN gathers exactly these planes
    net = ab.make_net(torch, 'cnn', board)
    x = torch.arange(ab.WIDTH, dtype=torch.float32).unsqueeze(0) + 1
    got = net.planes(x)[0].numpy()
    want = np.where(planes < ab.WIDTH, planes + 1, 0)
    assert np.array_equal(got, want)


def _dihedral(a, b):
    n = len(a)
    return any(all(b[k] == a[(r + s * k) % n] for k in range(n)) for r in range(n) for s in (1, -1))


def test_symmetry_then_tokenize_moves_whole_tiles(board):
    index = torch.from_numpy(ab.tile_index(board))
    x = torch.arange(ab.WIDTH).unsqueeze(0)
    tokens = x[:, index][0]
    perms = symmetry_index(torch)
    assert len(perms) == 12
    for p in perms:
        sym = x[:, p][:, index][0]
        sources = set()
        for u in range(19):
            t = int(p[8 * u]) // 8
            sources.add(t)
            assert torch.equal(sym[u, :8], tokens[t, :8])
            cs = [int(sym[u, 8 + 14 * k]) for k in range(6)]
            ct = [int(tokens[t, 8 + 14 * k]) for k in range(6)]
            es = [int(sym[u, 92 + 4 * k]) for k in range(6)]
            et = [int(tokens[t, 92 + 4 * k]) for k in range(6)]
            assert _dihedral(ct, cs) and _dihedral(et, es)
            assert sorted(sym[u].tolist()) == sorted(tokens[t].tolist())
        assert sources == set(range(19))


@pytest.mark.parametrize('arm', ab.ARMS)
def test_forward_gives_probabilities(board, arm):
    torch.manual_seed(0)
    net = ab.make_net(torch, arm, board).eval()
    with torch.no_grad():
        out = net(torch.randn(5, ab.WIDTH))
        win, lose = ab.log_p_win(torch, arm, out)
    if arm == 'heads':
        assert out[0].shape == (5, 4) and out[1].shape == (5,)
    else:
        assert out.shape == (5,)
    p = win.exp()
    assert p.shape == (5,) and torch.all((p > 0) & (p < 1))
    assert torch.allclose(p + lose.exp(), torch.ones(5), atol=1e-5)


def test_fast_auc_matches_reference():
    rng = random.Random(3)
    scores = [rng.choice([.1, .2, .3, .5, .9]) for _ in range(300)]
    labels = [int(rng.random() < .3 + .5 * s) for s in scores]
    assert abs(ab.fast_auc(scores, labels) - auc(scores, labels)) < 1e-12


def test_heads_targets_and_consistency_check(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    data_dir, runs = make_fixture(tmp_path)
    data = ab.shards(data_dir, 'train')
    recs = ab.load_records(runs, cfg, np.unique(data['game']), ab.manifests(data_dir, 'train')[1])
    rel, left, counts = ab.head_targets(data, recs)
    assert counts['truncated_games'] == 1 and counts['truncated_rows'] == 16
    assert np.all(rel[rel >= 0] < 4) and np.all(left >= 0)
    assert np.array_equal(rel == 0, data['y'] == 1)
    bad = dict(data, y=data['y'].copy())
    bad['y'][np.flatnonzero(rel == 1)[0]] = 1
    with pytest.raises(ValueError, match='consistency'):
        ab.head_targets(bad, recs)


def test_smoke_train_and_gate_all_arms(tmp_path):
    data, runs = make_fixture(tmp_path)
    out = tmp_path / 'out'
    common = ['--config', str(CONFIG), '--data', str(data), '--records', str(runs), '--out', str(out),
              '--device', 'cpu']
    for arm in ab.ARMS:
        result = ab.main(['train', '--arm', arm, '--epochs', '2', *common])
        assert result['smoke'] and len(result['history']) == 2 and result['chosen_epoch'] in (0, 1)
        assert (out / f'{arm}.pt').exists() and (out / f'{arm}-train.json').exists()
    with pytest.raises(FileExistsError):
        ab.main(['train', '--arm', 'base', '--epochs', '1', *common])
    gate = ab.main(['gate', *common])
    written = json.loads((out / 'gate.json').read_text())
    assert written['smoke'] and set(written['arms']) == set(ab.ARMS)
    for arm in ab.ARMS:
        m = written['arms'][arm]
        assert 0 <= m['auc'] <= 1 and m['log_loss'] > 0 and 0 <= m['ece10'] <= 1 and m['params'] > 0
        if arm != 'base':
            vs = m['vs_base']
            assert vs['ci95'][0] <= vs['ci95'][1] and isinstance(vs['pass'], bool)
    assert written['arms']['heads']['four_way_rows'] == gate['test_rows'] - 16
    assert written['bar'] == {'auc_diff_min': .01, 'lcb_gt': 0., 'min_diff': .01, 'resamples': 1000, 'seed': 29}
    with pytest.raises(FileExistsError):
        ab.main(['gate', *common])


def test_train_binds_verified_arrays_and_gate_checks_them(tmp_path):
    """Codex HOLD @b48f61d P1: the checkpoint carries the verified array hash; gate refuses
    a mismatch and labels a legacy checkpoint that has no binding."""
    data, runs = make_fixture(tmp_path)
    arrays = shards(data, 'train')['arrays_sha256']
    out = tmp_path / 'out'
    common = ['--config', str(CONFIG), '--data', str(data), '--records', str(runs), '--device', 'cpu']
    trained = ab.main(['train', '--arm', 'base', '--epochs', '1', '--out', str(out), *common])
    assert trained['train_arrays_sha256'] == arrays and trained['train_sealed_after_the_fact'] is None
    saved = torch.load(out / 'base.pt', weights_only=False)
    assert saved['train_arrays_sha256'] == arrays and saved['train_sealed_after_the_fact'] is None
    gate = ab.main(['gate', '--out', str(out), *common])
    assert gate['train_arrays_sha256'] == arrays and gate['arms']['base']['train_arrays_bound_in_checkpoint']
    assert gate['legacy_seal'] is False and gate['identity_limitations'] == []
    assert gate['test_arrays_sha256'] == shards(data, 'test')['arrays_sha256']

    wrong = tmp_path / 'wrong'
    wrong.mkdir()
    torch.save({**saved, 'train_arrays_sha256': '0' * 64}, wrong / 'base.pt')
    with pytest.raises(ValueError, match='different shard arrays'):
        ab.main(['gate', '--out', str(wrong), *common])

    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    torch.save({k: v for k, v in saved.items() if k not in ('train_arrays_sha256', 'train_sealed_after_the_fact')},
               legacy / 'base.pt')
    gate = ab.main(['gate', '--out', str(legacy), *common])
    assert gate['arms']['base']['train_arrays_bound_in_checkpoint'] is False
    assert len(gate['identity_limitations']) == 1 and 'predates array binding' in gate['identity_limitations'][0]

    changed = tmp_path / 'changed'
    changed.mkdir()
    torch.save({**saved, 'train_identity': '0' * 64}, changed / 'base.pt')
    with pytest.raises(ValueError, match='provenance'):
        ab.main(['gate', '--out', str(changed), *common])


def test_gate_refuses_tampered_train_arrays(tmp_path):
    data, runs = make_fixture(tmp_path)
    out = tmp_path / 'out'
    common = ['--config', str(CONFIG), '--data', str(data), '--records', str(runs), '--out', str(out),
              '--device', 'cpu']
    ab.main(['train', '--arm', 'base', '--epochs', '1', *common])
    npz = data / 'train-0000.npz'
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['y'] = 1 - arrays['y']
    with npz.open('wb') as stream:
        np.savez(stream, **arrays)
    with pytest.raises(ValueError, match='NPZ bytes differ'):
        ab.main(['gate', *common])
    assert not (out / 'gate.json').exists()

