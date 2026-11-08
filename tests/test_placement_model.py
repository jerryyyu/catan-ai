"""Placement model script: loader validations, extra-input wiring, loss, gate arithmetic, e2e."""
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
import placement_model as pm  # noqa: E402

CONFIG = ROOT / 'experiments' / 'placement-model.json'
LO = 2896000


def small_config(tmp, train=12, test=4, val=3):
    cfg = json.loads(CONFIG.read_text())
    cfg['data']['split'] = {'train': [LO, train], 'test': [LO + train, test]}
    cfg['data']['validation'] = (f'the last {val} train games ({LO + train - val}-{LO + train - 1}) choose the epoch; '
                                 'the test split is read once')
    path = tmp / 'config.json'
    path.write_text(json.dumps(cfg))
    return path, cfg


def make_game(seed, rng, decisions=2, candidates=(4, 3)):
    out = []
    for i in range(decisions):
        k = candidates[i % len(candidates)]
        post = rng.dirichlet(np.ones(4), size=3)
        cands = []
        for r in range(k):
            x = rng.normal(size=ab.WIDTH)
            p = 1 / (1 + math.exp(-x[0]))  # a little signal on one tile feature
            wins = int(rng.binomial(8, p))
            cands.append({'vertex': r, 'x': x.round(4).tolist(), 'wins': wins, 'playouts': 8,
                          'served': r == 0, 'served_rank': r if r < k - 1 else 20})
        out.append({'seat': i % 4, 'setup_index': (i * 3) % 8, 'style': 'builder',
                    'opponent_styles': ['tactical', 'builder', 'expander'], 'posterior': post.tolist(),
                    'chosen_vertex': 0, 'candidates': cands})
    return {'seed': seed, 'lineup': ['builder'] * 4, 'winner': 0, 'turns': 50, 'seconds': 1., 'decisions': out}


def make_data(tmp, cfg, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    root = tmp / 'data'
    (root / 'games').mkdir(parents=True)
    (root / 'config.json').write_text(json.dumps({'config': {'split': cfg['data']['split']}}))
    for lo, n in cfg['data']['split'].values():
        for s in range(lo, lo + n):
            (root / 'games' / f'{s}.json').write_text(json.dumps(make_game(s, rng)))
    return root


def edit_game(root, seed, fn):
    path = root / 'games' / f'{seed}.json'
    g = json.loads(path.read_text())
    fn(g)
    path.write_text(json.dumps(g))


# ------------------------------------------------------------------ loader
def test_loader_shapes_and_counts(tmp_path):
    _, cfg = small_config(tmp_path)
    root = make_data(tmp_path, cfg)
    d = pm.load(root, cfg, 'test')
    assert d['x'].shape == (4 * 7, ab.WIDTH) and d['extra'].shape == (28, pm.EXTRA_W)
    assert d['candidate_counts'] == {'3': 4, '4': 4}
    assert list(np.diff(d['start'])) == [4, 3] * 4
    e = d['extra'][0]
    assert np.isclose(e[:12].reshape(3, 4).sum(1), 1).all() and e[12:20].sum() == 1 and e[20] == 0
    assert d['extra'][3, 20] == 20 / 16  # served_rank / 16, literal (not clipped)


def test_loader_refuses_file_outside_split(tmp_path):
    _, cfg = small_config(tmp_path)
    root = make_data(tmp_path, cfg)
    (root / 'games' / f'{LO + 999}.json').write_text(json.dumps(make_game(LO + 999, np.random.default_rng(1))))
    with pytest.raises(ValueError, match='outside the declared split'):
        pm.load(root, cfg, 'train')


def test_loader_refuses_seed_name_mismatch(tmp_path):
    _, cfg = small_config(tmp_path)
    root = make_data(tmp_path, cfg)
    edit_game(root, LO + 13, lambda g: g.update(seed=LO + 14))
    with pytest.raises(ValueError, match='holds seed'):
        pm.load(root, cfg, 'test')


def test_loader_refuses_served_rank_not_unique(tmp_path):
    _, cfg = small_config(tmp_path)
    root = make_data(tmp_path, cfg)
    edit_game(root, LO + 12, lambda g: g['decisions'][1]['candidates'][1].update(served_rank=0))
    with pytest.raises(ValueError, match='served_rank 0'):
        pm.load(root, cfg, 'test')
    edit_game(root, LO + 12, lambda g: [c.update(served_rank=c['served_rank'] + 1)
                                        for c in g['decisions'][1]['candidates']])
    with pytest.raises(ValueError, match='served_rank 0'):
        pm.load(root, cfg, 'test')


def test_loader_refuses_bad_posterior_and_width(tmp_path):
    _, cfg = small_config(tmp_path)
    root = make_data(tmp_path, cfg)
    edit_game(root, LO + 12, lambda g: g['decisions'][0].update(posterior=[[.5, .5, .5, 0]] * 3))
    with pytest.raises(ValueError, match='posterior'):
        pm.load(root, cfg, 'test')
    root2 = make_data(tmp_path / 'b', cfg)
    edit_game(root2, LO + 15, lambda g: g['decisions'][0]['candidates'][0].update(x=[0.] * 10))
    with pytest.raises(ValueError, match='length'):
        pm.load(root2, cfg, 'test')


def test_smoke_seed_selection_keeps_validation():
    cfg = json.loads(CONFIG.read_text())
    seeds = pm.selected_seeds(cfg, 'train', 60)
    assert len(seeds) == 60 and seeds[:54] == list(range(LO, LO + 54)) and seeds[54:] == list(range(LO + 405, LO + 411))
    assert pm.validation_games(cfg) == 45
    assert pm.gate_constants(cfg) == {'lcb_gt': 0., 'resamples': 1000, 'seed': 29}
    assert len(pm.selected_seeds(cfg, 'train')) == 450


# ------------------------------------------------------------------- model
def test_extra_inputs_and_tiles_reach_the_output():
    torch.manual_seed(0)
    net = pm.make_model(torch).eval()
    x = torch.randn(3, ab.WIDTH)
    e = torch.rand(3, pm.EXTRA_W)
    base = net(x, e)
    e2 = e.clone()
    e2[1, :4] = torch.tensor([1., 0, 0, 0])
    out = net(x, e2)
    assert out[1] != base[1] and torch.equal(out[0], base[0])
    x2 = x.clone()
    x2[2, int(net.core.tidx[5, 0])] += 3.  # one tile token's first feature
    out = net(x2, e)
    assert out[2] != base[2] and torch.equal(out[0], base[0])
    assert net.core.glob.in_features == ab.GLOBAL_W + pm.EXTRA_W


# -------------------------------------------------------------------- loss
def test_listwise_hand_computed():
    logits = torch.tensor([0., math.log(2.), 1., 2., 3.], dtype=torch.float64)
    rate = torch.tensor([.25, .75, 0., 0., 0.], dtype=torch.float64)
    idx, mask = pm.decision_index(torch, np.array([0, 2, 5]), 'cpu')
    terms, valid = pm.listwise_terms(torch, logits, rate, idx, mask)
    want = -(.25 * math.log(1 / 3) + .75 * math.log(2 / 3))
    assert valid.tolist() == [True, False]
    assert terms[0].item() == pytest.approx(want, abs=1e-12)
    total, bce, lw, nv = pm.losses(torch, logits, rate, idx, mask)
    sig = 1 / (1 + np.exp(-logits.numpy()))
    r = rate.numpy()
    want_bce = -np.mean(r * np.log(sig) + (1 - r) * np.log(1 - sig))
    assert nv == 1 and lw.item() == pytest.approx(want) and bce.item() == pytest.approx(want_bce)
    assert total.item() == pytest.approx(want_bce + want)


# -------------------------------------------------------------------- gate
def tiny_gate_data():
    # 3 games x 1 decision x 3 candidates; served is the candidate with rank 0
    rate = np.array([.5, .25, 0., .125, .375, .625, 0., 0., .25])
    rank = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0])
    return {'rate': rate, 'rank': rank, 'wins': (rate * 8).astype(int), 'playouts': np.full(9, 8),
            'start': np.array([0, 3, 6, 9]), 'dec_game': np.array([10, 11, 12]),
            'dec_seat': np.array([0, 1, 0]), 'dec_setup': np.array([0, 5, 2]), 'game': np.repeat([10, 11, 12], 3)}


def test_gate_arithmetic_tiny_example():
    data = tiny_gate_data()
    logits = np.array([0., 1., -1., 2., 0., 1., 0., 3., 1.])
    # picks: d0 -> idx1 (.25) vs served .5; d1 -> idx3 (.125) vs served .375; d2 -> idx7 (0) vs served .25
    st = pm.decision_stats(data, logits)
    assert list(st['pick'] - st['served']) == [-.25, -.25, -.25]
    logits = np.array([3., 1., -1., 0., 0., 1., 2., 0., 1.])
    st = pm.decision_stats(data, logits)
    diffs = st['pick'] - st['served']
    assert list(diffs) == [0., .25, -.25]
    assert list(st['best'] - st['pick']) == [0., 0., .25]
    assert list(st['pick_unique_best']) == [1., 1., 0.]
    # bootstrap by hand with the same rng contract: resample game ids with replacement
    rng, ids, by = random.Random(29), [10, 11, 12], {10: 0., 11: .25, 12: -.25}
    stats = sorted(float(np.mean([by[rng.choice(ids)] for _ in ids])) for _ in range(40))
    point, ci = pm.bootstrap(diffs, data['dec_game'], 40, 29)
    assert point == pytest.approx(0.) and ci == [stats[1], stats[38]]
    report = pm.gate_report(data, logits, {'lcb_gt': 0., 'resamples': 40, 'seed': 29}, .3)
    assert report['primary']['pass'] is False and report['primary']['lcb'] == stats[1]
    assert report['by_placement']['first']['pick_minus_served'] == pytest.approx((0. - .25) / 2)
    assert report['by_placement']['second']['pick_minus_served'] == pytest.approx(.25)
    assert report['baselines']['random']['pick_minus_served'] == pytest.approx(
        np.mean([.25 - .5, .375 - .375, .25 / 3 - .25]))
    assert report['model']['spearman_mean'] == pytest.approx(np.mean([1., math.sqrt(3) / 2, 0.]))  # midranks: d1 scores tie, d2 rates tie
    assert report['model']['spearman_decisions'] == 3


# --------------------------------------------------------------------- e2e
def test_train_and_gate_end_to_end(tmp_path):
    cfg_path, cfg = small_config(tmp_path)
    root = make_data(tmp_path, cfg)
    out = tmp_path / 'out'
    common = ['--config', str(cfg_path), '--data', str(root), '--out', str(out), '--device', 'cpu']
    tr = pm.main(['train', *common, '--epochs', '2'])
    assert tr['smoke'] and len(tr['history']) == 2 and tr['chosen_epoch'] in (0, 1)
    assert tr['fit_games'] == 9 and tr['val_games'] == 3 and tr['rows'] == 12 * 7
    assert (out / 'model.pt').exists() and (out / 'train.json').exists()
    with pytest.raises(FileExistsError):
        pm.main(['train', *common, '--epochs', '2'])
    with pytest.raises(ValueError, match='smoke'):
        pm.main(['gate', *common])
    g = pm.main(['gate', *common, '--allow-smoke'])
    saved = json.loads((out / 'gate.json').read_text())
    for key in ('primary', 'rates', 'by_setup_index', 'by_placement', 'by_seat', 'model', 'baselines',
                'test_identity', 'train_identity', 'model_sha256', 'config_sha256', 'script_sha256', 'seconds'):
        assert key in saved
    assert saved['primary']['decisions'] == 8 and saved['primary']['games'] == 4 and g['smoke']
    assert set(saved['baselines']) == {'served', 'random'}
    # a changed training file breaks the provenance check
    edit_game(root, LO, lambda g: g.update(turns=51))
    (out / 'gate.json').unlink()
    with pytest.raises(ValueError, match='train identity'):
        pm.main(['gate', *common, '--allow-smoke'])
