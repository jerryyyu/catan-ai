"""tiles_margin (experiments/tiles-margin.json): final-points replay, margin targets, loss,
win-head export equivalence, a synthetic train/win/gate smoke and the planner registration."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from catan_ai.experiments import source_files
from catan_ai.parameter_planner import DEFAULT

ROOT = Path(__file__).resolve().parent.parent
replay = runpy.run_path(str(ROOT / 'scripts' / 'replay_final_points.py'))
rank = runpy.run_path(str(ROOT / 'scripts' / 'rank_data.py'))
CONFIG = ROOT / 'experiments' / 'tiles-margin.json'
BASE = ROOT / 'experiments' / 'arch-bakeoff.json'


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


# ------------------------------------------------------------------ replay (no numpy)
def test_compare_reports_match_and_each_kind_of_mismatch():
    stored = {'winner': 2, 'turns': 70, 'truncated_game': False,
              'value_rows': [{'turn': 1, 'seat': 0, 'point': 'roll'}]}
    same = {'points': [9, 8, 10, 4], 'public_points': [6, 6, 10, 4], 'winner': 2, 'turns': 70}
    r = replay['compare'](5, stored, same, [[1, 0, 'roll']])
    assert r['match'] and r['value_rows_match'] and r['winner_has_max_points']
    assert r['hidden_vp'] == [3, 2, 0, 0] and r['points'] == [9, 8, 10, 4]
    assert not replay['compare'](5, stored, dict(same, turns=71), [[1, 0, 'roll']])['match']
    assert not replay['compare'](5, stored, dict(same, winner=1), [[1, 0, 'roll']])['match']
    rows_only = replay['compare'](5, stored, same, [[1, 1, 'roll']])
    assert rows_only['match'] and not rows_only['value_rows_match']  # reported, not part of match
    s = replay['summarize']([r, replay['compare'](6, stored, dict(same, turns=71), [])])
    assert s['games'] == 2 and s['matches'] == 1
    assert s['mismatches'] == [{'seed': 6, 'winner': 2, 'turns': 71, 'stored_winner': 2, 'stored_turns': 70}]
    assert s['value_rows_mismatches'] == [6]


def _root(tmp_path, seeds=(2650000,)):
    config = {'name': 'tiny', 'schema': 2, 'policy': 'planner_offer_accept', 'max_turns': 1000,
              'seeds': [seeds[0], len(seeds)], 'decision_rate': 0.0, 'candidates': 4, 'playouts': 0,
              'value_points': ['roll', 'end_turn']}
    root = tmp_path / 'data'
    identity = rank['bind_manifest'](root, rank['run_identity'](config))
    (root / 'games').mkdir()
    for s in seeds:
        rank['run_game']((s, config, str(root / 'games' / f'{s}.json'), identity['identity_sha256']))
    return root


def test_replay_reproduces_a_producer_game_and_flags_a_tampered_record(tmp_path):
    root = _root(tmp_path)
    stored = json.loads((root / 'games' / '2650000.json').read_text())
    out = tmp_path / 'fp'
    summary = replay['main'](['--root', str(root), '--out', str(out)])
    assert summary['games'] == summary['matches'] == 1 and not summary['mismatches']
    got = json.loads((out / 'games' / '2650000.json').read_text())
    assert got['winner'] == stored['winner'] and got['turns'] == stored['turns'] and got['value_rows_match']
    assert got['points'][got['winner']] >= 10 and max(p for i, p in enumerate(got['points']) if i != got['winner']) < 10
    assert got['record_sha256'] == hashlib.sha256((root / 'games' / '2650000.json').read_bytes()).hexdigest()
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['root_identity_sha256'] == json.loads((root / 'manifest.json').read_text())['identity_sha256']
    assert manifest['source_drift'] == []
    # a second run into the same --out reuses the written game (open('x') never clobbers)
    again = replay['main'](['--root', str(root), '--out', str(out), '--split', '2650000', '2650001'])
    assert again['matches'] == 1
    # a stored record whose turn count differs is reported as a mismatch, not repaired
    tampered = dict(stored, turns=stored['turns'] + 1)
    (root / 'games' / '2650000.json').write_text(json.dumps(tampered))
    bad = replay['main'](['--root', str(root), '--out', str(tmp_path / 'fp2')])
    assert bad['matches'] == 0 and bad['mismatches'][0]['stored_turns'] == stored['turns'] + 1


def test_replay_refuses_foreign_sources_and_records(tmp_path):
    root = _root(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['source']['engine.py'] = '0' * 64
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='differs from the producer sources'):
        replay['main'](['--root', str(root), '--out', str(tmp_path / 'fp')])
    record = json.loads((root / 'games' / '2650000.json').read_text())
    (root / 'games' / '2650000.json').write_text(json.dumps(dict(record, identity='x')))
    with pytest.raises(ValueError, match='does not belong'):
        replay['main'](['--root', str(root), '--out', str(tmp_path / 'fp3'), '--allow-source-drift'])


# ------------------------------------------------------------------ numpy / torch
def _modules():
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'tests'))
    import arch_bakeoff as ab
    import tiles_margin as tm
    return np, torch, ab, tm


def test_declared_constants_are_read_from_the_declaration():
    _, _, _, tm = _modules()
    cfg, base = json.loads(CONFIG.read_text()), json.loads(BASE.read_text())
    tm.check_configs(cfg, base)
    assert tm.margin_constants(cfg) == (.5, 1.)
    assert tm.gate_constants(cfg) == {'reference_auc': .7859, 'min_diff': 0., 'lcb_gt': -.002,
                                      'resamples': 1000, 'seed': 29, 'xeval_min': .0062}


def test_margin_targets_use_the_seat_one_hot_and_exclude_mismatches():
    np, _, ab, tm = _modules()
    rows = [(7, 0), (7, 2), (7, 3), (8, 1), (8, 2), (9, 0)]  # (game, seat)
    final = {7: {'points': [10, 6, 8, 3], 'winner': 0, 'match': True},
             8: {'points': [4, 10, 9, 7], 'winner': 1, 'match': False},
             9: {'points': [5, 5, 5, 5], 'winner': None, 'match': True}}
    x = np.zeros((len(rows), ab.WIDTH), dtype=np.float16)
    for i, (_, seat) in enumerate(rows):
        x[i, ab.SEAT_COLS.start + seat] = 1
    data = {'x': x, 'game': np.array([g for g, _ in rows]),
            'y': np.array([int(final[g]['winner'] == s) for g, s in rows], dtype=np.uint8)}
    target, mask, counts = tm.margin_targets(data, final)
    assert mask.tolist() == [True, True, True, False, False, False]
    assert np.allclose(target, [.2, -.2, -.7, 0, 0, 0])
    assert counts['margin_rows'] == 3 and counts['excluded_rows'] == 3
    assert counts['mismatch_games'] == 1 and counts['mismatch_seeds'] == [8] and counts['truncated_games'] == 1
    data['y'][1] = 1  # a label that disagrees with the replayed winner
    with pytest.raises(ValueError, match='consistency'):
        tm.margin_targets(data, final)


def test_loss_is_mean_bce_plus_half_mean_huber_over_margin_rows():
    _, torch, _, tm = _modules()
    w = torch.tensor([0., 2., -1.])
    m = torch.tensor([.5, 3., 9.])
    y = torch.tensor([1., 0., 1.])
    t = torch.tensor([.1, .5, 0.])
    mask = torch.tensor([True, True, False])
    loss, bce, hub, k = tm.batch_loss(torch, w, m, y, t, mask, .5, 1.)
    import math
    want_bce = math.log(2) + math.log(1 + math.e ** 2) + math.log(1 + math.e)
    want_hub = .5 * .4 ** 2 + (2.5 - .5)  # quadratic below delta 1, linear above
    assert k == 2 and abs(bce.item() - want_bce) < 1e-5 and abs(hub.item() - want_hub) < 1e-5
    assert abs(loss.item() - (want_bce / 3 + .5 * want_hub / 2)) < 1e-5
    loss0, _, hub0, k0 = tm.batch_loss(torch, w, m, y, t, torch.zeros(3, dtype=torch.bool), .5, 1.)
    assert k0 == 0 and hub0.item() == 0 and abs(loss0.item() - want_bce / 3) < 1e-5


def test_win_head_of_the_margin_model_equals_a_plain_tiles_model_without_the_margin_head():
    _, torch, ab, tm = _modules()
    torch.manual_seed(1)
    board = ab.layout()
    net = tm.make_margin_net(torch, board).eval()
    plain = ab.make_net(torch, 'tiles', board)
    state = tm.win_state(net.state_dict())
    assert not any(k.startswith('margin') for k in state)
    plain.load_state_dict(state)  # strict: the same keys as an arch-bakeoff tiles model
    plain.eval()
    x = torch.randn(7, ab.WIDTH)
    with torch.no_grad():
        win, margin = net(x)
        assert margin.shape == (7,)
        assert torch.allclose(win, plain(x), atol=1e-6)
    with pytest.raises(ValueError):
        tm.win_state(plain.state_dict())


def write_final_points(np, data_dir, runs, mismatch_seed=None, seed=0):
    """Final-points replay outputs for every game record of a make_fixture run, consistent with
    the records (winner gets 10, others below 10)."""
    rng = np.random.default_rng(seed)
    out = runs / 'final-points'
    for split in ('train', 'test'):
        sources = {}
        for f in sorted(Path(data_dir).glob(f'{split}-*.json')):
            sources.update(json.loads(f.read_text())['source_sha256'])
        for root in ('value100k-v1', 'value100k-v1b'):
            d = out / root / 'games'
            d.mkdir(parents=True, exist_ok=True)
            (d.parent / 'manifest.json').write_text(json.dumps({'manifest_sha256': f'm-{root}'}))
            for path in sorted((runs / root / 'games').glob('*.json')):
                rec = json.loads(path.read_text())
                s = rec['seed']
                if str(s) not in sources or (d / f'{s}.json').exists():
                    continue
                pts = [int(v) for v in rng.integers(2, 10, 4)]
                if rec['winner'] is not None:
                    pts[rec['winner']] = 10
                (d / f'{s}.json').write_text(json.dumps({
                    'seed': s, 'points': pts, 'winner': rec['winner'], 'turns': rec['turns'],
                    'match': s != mismatch_seed, 'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'manifest_sha256': f'm-{root}'}))
    return out


def test_final_points_must_come_from_the_shard_source_records(tmp_path):
    np, _, ab, tm = _modules()
    from test_arch_bakeoff import make_fixture
    data_dir, runs = make_fixture(tmp_path)
    fp = write_final_points(np, data_dir, runs)
    data = ab.shards(data_dir, 'train')
    sources = ab.manifests(data_dir, 'train')[1]
    final, binding = tm.load_final_points(fp, np.unique(data['game']), sources)
    assert set(final) == {int(g) for g in np.unique(data['game'])} and set(binding) == {'value100k-v1', 'value100k-v1b'}
    g = int(data['game'][0])
    with pytest.raises(ValueError, match='different game record'):
        tm.load_final_points(fp, [g], {**sources, g: 'f' * 64})


def test_synthetic_train_win_xeval_binding_and_gate_end_to_end(tmp_path):
    np, torch, ab, tm = _modules()
    from test_arch_bakeoff import make_fixture
    data_dir, runs = make_fixture(tmp_path)
    data = ab.shards(data_dir, 'train')
    mismatch = int(np.unique(data['game'])[2])
    fp = write_final_points(np, data_dir, runs, mismatch_seed=mismatch)
    ref_out, out = tmp_path / 'ref', tmp_path / 'out'
    ab.main(['train', '--config', str(BASE), '--arm', 'tiles', '--data', str(data_dir), '--out', str(ref_out),
             '--device', 'cpu', '--epochs', '2'])
    common = ['--config', str(CONFIG), '--base-config', str(BASE), '--out', str(out), '--device', 'cpu']
    trained = tm.main(['train', *common, '--data', str(data_dir), '--final-points', str(fp), '--epochs', '2'])
    assert trained['smoke'] and trained['epochs'] == 2 and len(trained['history']) == 2
    assert trained['margin_counts']['mismatch_seeds'] == [mismatch] and trained['margin_counts']['excluded_rows'] > 0
    assert trained['train_arrays_sha256'] == data['arrays_sha256']
    assert all(e['val_margin']['rows'] > 0 for e in trained['history'])
    with pytest.raises(FileExistsError):
        tm.main(['train', *common, '--data', str(data_dir), '--final-points', str(fp), '--epochs', '2'])
    win = tm.main(['win', *common])
    saved = torch.load(out / 'tiles_margin-win.pt', weights_only=False)
    assert saved['arm'] == 'tiles' and saved['derived_from']['dropped'] == ['margin.bias', 'margin.weight']
    assert saved['params'] == trained['params'] - (ab.TILES_D + 1)
    # the gate binds an xeval result to this checkpoint's win-head file
    xeval = {'model_sha256': win['win_sha256'], 'pick_minus_served': .01, 'pick_minus_served_lcb95': .001,
             'eligible_for_next_gate': True}
    (tmp_path / 'xeval.json').write_text(json.dumps(xeval))
    gated = tm.main(['gate', *common, '--data', str(data_dir), '--final-points', str(fp),
                     '--reference', str(ref_out / 'tiles.pt'), '--xeval', str(tmp_path / 'xeval.json')])
    assert gated['smoke'] and gated['missing'] == [] and gated['xeval_pass'] is True
    assert gated['pass'] == (gated['vs_reference']['pass'] and gated['xeval_pass'])
    for arm in ('tiles_margin', 'tiles_reference'):
        assert {'auc', 'log_loss', 'ece10'} <= set(gated['arms'][arm])
    assert gated['arms']['tiles_margin']['margin_head_test']['rows'] > 0
    (tmp_path / 'xeval.json').write_text(json.dumps(dict(xeval, model_sha256='0' * 64)))
    (out / 'gate.json').unlink()
    with pytest.raises(ValueError, match='xeval was not run'):
        tm.main(['gate', *common, '--data', str(data_dir), '--reference', str(ref_out / 'tiles.pt'),
                 '--xeval', str(tmp_path / 'xeval.json')])


def _synthetic_export(np, torch, ab, tm, tmp_path):
    """A random (non-smoke) tiles_margin checkpoint -> win checkpoint -> export_tiles_npz."""
    import export_tiles_npz as ex
    torch.manual_seed(4)
    out = tmp_path / 'run'
    out.mkdir()
    net = tm.make_margin_net(torch, ab.layout())
    cfg = json.loads(CONFIG.read_text())
    torch.save({'arm': tm.ARM, 'state': net.state_dict(), 'mean': torch.zeros(ab.WIDTH), 'sd': torch.ones(ab.WIDTH),
                'epoch': 0, 'val_bce': .5, 'config': cfg, 'train_identity': 'i', 'train_arrays_sha256': 'a',
                'train_sealed_after_the_fact': None, 'train_seconds': 1., 'smoke': False, 'max_rows': None,
                'epochs': 12, 'params': sum(p.numel() for p in net.parameters())}, out / 'tiles_margin.pt')
    tm.main(['win', '--config', str(CONFIG), '--base-config', str(BASE), '--out', str(out)])
    stem = tmp_path / 'value_tiles_margin_v1'
    ex.main(['--pt', str(out / 'tiles_margin-win.pt'), '--npz', f'{stem}.npz', '--meta', f'{stem}.meta.json',
             '--source-run', 'synthetic'])
    return net, stem


def test_export_of_the_win_head_serves_the_margin_models_win_logit(tmp_path, monkeypatch):
    np, torch, ab, tm = _modules()
    from catan_ai import pv_search, tiles_value as tv
    from catan_ai.bots import make_policy
    net, stem = _synthetic_export(np, torch, ab, tm, tmp_path)
    meta = json.loads(Path(f'{stem}.meta.json').read_text())
    assert meta['source_run'] == 'synthetic' and meta['derived_from']['arm'] == 'tiles_margin'
    assert meta['parity_max_abs_diff'] < tv.PARITY_ATOL and meta['params'] == 555009
    monkeypatch.setattr(pv_search, 'TILES_MARGIN_V1', stem)
    bot = make_policy('planner_pv2_tiles_margin', 1)
    assert isinstance(bot, pv_search.PVSearchTilesMarginPlanner) and isinstance(bot, pv_search.PVSearchTilesPlanner)
    assert pv_search.PVSearchTilesMarginPlanner.choose is pv_search.PVSearchTilesPlanner.choose
    assert pv_search.PVSearchTilesMarginPlanner._logits is pv_search.PVSearchTilesPlanner._logits
    x = np.random.default_rng(6).standard_normal((3, ab.WIDTH))
    with torch.no_grad():
        want = net.eval()(torch.from_numpy(x.astype(np.float32)))[0].double().numpy()
    assert np.allclose(bot._logits(list(x)), want, atol=1e-4)


def test_planner_pv2_tiles_margin_loads_the_real_weights_when_present():
    pytest.importorskip('numpy')
    from catan_ai import pv_search
    from catan_ai.bots import make_policy
    if not pv_search.TILES_MARGIN_V1.with_suffix('.npz').exists():
        with pytest.raises(FileNotFoundError):
            make_policy('planner_pv2_tiles_margin', 1)
        pytest.skip('tiles_margin weights are a local artifact (not committed)')
    bot = make_policy('planner_pv2_tiles_margin', 1)
    assert bot.net.meta['derived_from']['arm'] == 'tiles_margin'
