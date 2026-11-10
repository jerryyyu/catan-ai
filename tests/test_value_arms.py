import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
DECL = json.loads((ROOT / 'experiments' / 'value-arms.json').read_text())


def test_declaration_is_consistent():
    served = json.loads((ROOT / 'checkpoints' / 'value_tiles_v1.meta.json').read_text())['pt_sha256']
    assert DECL['reference']['pt_sha256'] == served
    gate = DECL['gate']
    classes = gate['capacity_arms']['arms'] + gate['support_arms']['arms']
    assert sorted(classes) == sorted(DECL['arms'])
    for spec in DECL['arms'].values():
        assert all(r in DECL['roots'] for r in spec['data'])
    for root in DECL['roots'].values():
        assert (ROOT / 'experiments' / root['declaration']).exists()
    base = json.loads((ROOT / 'experiments' / 'value-v2-100k.json').read_text())
    for name in ('value-v2-100k-roll.json', 'endnow-v1.json'):
        d = json.loads((ROOT / 'experiments' / name).read_text())
        assert d['roots'] == base['roots'] and d['split'] == base['split']


def test_verdict_classes():
    import value_arms_gate as g
    gate = DECL['gate']
    good = {'original': {'lcb_1pct': -.001}, 'onpolicy': {'lcb_1pct': .001}}
    xe = {'eligible_for_next_gate': True, 'pick_minus_served': .008, 'city_ranked_above_end_by_model': .75}
    xr = {'pick_minus_served': .0085}
    assert g.verdict('t9', gate, good, xe, xr, []) == (
        {'original_auc': True, 'onpolicy_auc': True, 'xeval': True, 'onpolicy_gain': True}, True)
    assert g.verdict('s2', gate, good, xe, xr, [])[1] is True
    assert g.verdict('s2', gate, good, dict(xe, city_ranked_above_end_by_model=.9), xr, [])[1] is False
    flat = {'original': {'lcb_1pct': -.001}, 'onpolicy': {'lcb_1pct': -.001}}
    assert g.verdict('t7', gate, flat, xe, xr, [])[1] is False   # capacity arms must gain on-policy
    assert g.verdict('s1', gate, flat, xe, xr, [])[1] is True    # support arms need only non-inferiority
    assert g.verdict('t9', gate, good, xe, xr, ['new: script differs'])[1] is None
    assert g.verdict('t9', gate, {'original': {'lcb_1pct': -.006}, 'onpolicy': {'lcb_1pct': .01}}, xe, xr, [])[1] is False


def test_endnow_fork_is_deterministic_and_labelled():
    import endnow_data as e
    a, b = e.play(2650000, 'planner_offer_accept', 1000), e.play(2650000, 'planner_offer_accept', 1000)
    assert a[0] is not None and a[2] > 0
    assert a[0][1] in (0, 1) and a[0][1:] == b[0][1:] and list(a[0][0]) == list(b[0][0])


torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
import arch_bakeoff as ab  # noqa: E402


def test_make_net_defaults_build_the_served_net():
    net = ab.make_net(torch, 'tiles')
    assert sum(p.numel() for p in net.parameters()) == 555009
    assert not any('phase' in k for k in net.state_dict())
    assert sorted(net.state_dict()) == sorted(ab.make_net(torch, 'tiles', size={}).state_dict())
    big = ab.make_net(torch, 'tiles', size=DECL['arms']['t7']['size'])
    assert 2_500_000 < sum(p.numel() for p in big.parameters()) < 4_000_000
    with pytest.raises(ValueError):
        ab.make_net(torch, 'base', size={'d': 64})


def test_phase_zero_matches_the_unflagged_forward():
    torch.manual_seed(0)
    net = ab.make_net(torch, 'tiles', phase=True).eval()
    with torch.no_grad():
        net.phase_vec.normal_()
        x = torch.randn(5, ab.WIDTH)
        assert torch.equal(net(x), net(x, torch.zeros(5)))
        assert not torch.allclose(net(x), net(x, torch.ones(5)))
    with pytest.raises(ValueError):
        ab.make_net(torch, 'tiles')(x, torch.zeros(5))


def _root(tmp, name, lo, n, phase_rows=3):
    import reencode_v2
    out = tmp / f'{name}-shards'
    out.mkdir()
    decl = {'name': name, 'roots': [{'root': name}], 'point': name, 'stride': 1, 'max_turns': 1000,
            'shard_games': n, 'split': {'train': [lo, n], 'test': [lo + n, 4]}}
    rng = np.random.default_rng(lo)
    for split, (a, k) in decl['split'].items():
        res = [(s, [rng.normal(size=ab.WIDTH).astype(np.float32) for _ in range(phase_rows)],
                [int(rng.random() < .3) for _ in range(phase_rows)], [(1, 0, 0., 0.)] * phase_rows, 'h', False)
               for s in range(a, a + k)]
        reencode_v2.write_shard(out, split, 0, list(range(a, a + k)), res)
    (out / 'DONE.json').write_text(json.dumps({'config': decl}))
    path = tmp / f'{name}.json'
    path.write_text(json.dumps(decl))
    return {'shards': str(out), 'declaration': str(path)}


def test_train_smoke_mixes_roots_with_phase_and_weight(tmp_path):
    import value_arms as va
    cfg = json.loads(json.dumps(DECL))
    cfg['roots'] = {'original': _root(tmp_path, 'a', 100, 12), 'roll': _root(tmp_path, 'b', 300, 12)}
    cfg['selection'] = ['original']
    cfg['common'].update(validation_games=3, batch=8)
    cfg['arms'] = {'s1': {'data': ['original', 'roll'], 'phase': {'roll': 1}, 'weight': {'roll': 2.0}}}
    conf = tmp_path / 'arms.json'
    conf.write_text(json.dumps(cfg))
    out = va.main(['--config', str(conf), '--arm', 's1', '--runs', '/', '--out', str(tmp_path / 'o'),
                   '--device', 'cpu', '--epochs', '1'])
    assert out['fit_rows'] == 2 * 9 * 3 and out['selection_val_rows'] == 3 * 3 and out['smoke']
    saved = torch.load(tmp_path / 'o' / 's1.pt', weights_only=False)
    assert 'phase_vec' in saved['state'] and saved['value_arm'] == 's1'
    bad = dict(cfg, roots={**cfg['roots'], 'roll': dict(cfg['roots']['roll'], declaration=cfg['roots']['original']['declaration'])})
    conf.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='not made under'):
        va.main(['--config', str(conf), '--arm', 's1', '--runs', '/', '--out', str(tmp_path / 'p'), '--device', 'cpu', '--epochs', '1'])


def test_gate_binds_every_training_root(tmp_path):
    """Codex HOLD @e19e5a1 (1): the recorded root set and each root's identity, digest and seal must match."""
    import value_arms as va
    cfg = json.loads(json.dumps(DECL))
    cfg['roots'] = {'original': _root(tmp_path, 'a', 100, 12), 'roll': _root(tmp_path, 'b', 300, 12)}
    cfg['arms'] = {'s1': {'data': ['original', 'roll'], 'phase': {'roll': 1}}}
    rec = {n: va.root_identity(r['shards']) for n, r in cfg['roots'].items()}
    va.check_training(cfg, 's1', {'train': rec}, '/')
    with pytest.raises(ValueError, match='records roots'):
        va.check_training(cfg, 's1', {'train': {'original': rec['original']}}, '/')  # auxiliary root missing
    for name in ('original', 'roll'):
        for key in ('identity', 'arrays_sha256', 'done_sha256'):
            bad = {n: dict(v) for n, v in rec.items()}
            bad[name][key] = 'stale'
            with pytest.raises(ValueError, match=f'{name} {key}'):
                va.check_training(cfg, 's1', {'train': bad}, '/')


def test_train_receipt_and_checkpoint_carry_producer_provenance(tmp_path):
    """Codex HOLD @e19e5a1 (2)."""
    import value_arms as va
    cfg = json.loads(json.dumps(DECL))
    cfg['roots'] = {'original': _root(tmp_path, 'a', 100, 12), 'roll': _root(tmp_path, 'b', 300, 12)}
    cfg['selection'] = ['original']
    cfg['common'].update(validation_games=3, batch=8)
    cfg['arms'] = {'t9': {'data': ['original', 'roll']}}
    conf = tmp_path / 'arms.json'
    conf.write_text(json.dumps(cfg))
    out = va.main(['--config', str(conf), '--arm', 't9', '--runs', '/', '--out', str(tmp_path / 'o'), '--device', 'cpu', '--epochs', '1'])
    saved = torch.load(tmp_path / 'o' / 't9.pt', weights_only=False)
    for prov in (out['provenance'], saved['provenance']):
        assert prov['source_revision'] and prov['engine_backend'] in ('python', 'cython')
        assert 'engine_backend' in prov['source_files'] and 'encoding.py' in prov['source_files']
        assert {'value_arms.py', 'arch_bakeoff.py', 'train_value_v2.py'} <= set(prov['scripts_sha256'])
        assert 'torch' in prov['versions']
    assert all('done_sha256' in v for v in out['train'].values())


def test_endnow_done_seals_producer_provenance(tmp_path):
    import endnow_data as e
    seed = 2650000
    row, _, _ = e.play(seed, 'planner_offer_accept', 1000)
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    game = Game.new([f'p{i}' for i in range(4)], seed)
    bots = [make_policy('planner_offer_accept', seed * 17 + i + 1) for i in range(4)]
    while game.winner is None:
        s = game.current
        game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    src = tmp_path / 'runs' / 'src'
    (src / 'games').mkdir(parents=True)
    (src / 'manifest.json').write_text(json.dumps({'identity_sha256': 'idx', 'config': {'policy': 'planner_offer_accept'}}))
    (src / 'games' / f'{seed}.json').write_text(json.dumps({'seed': seed, 'identity': 'idx', 'winner': game.winner}))
    decl = {'roots': [{'root': 'src', 'identity_sha256': 'idx'}], 'point': 'endnow', 'stride': 1, 'max_turns': 1000,
            'shard_games': 1, 'split': {'train': [seed, 1]}}
    conf = tmp_path / 'endnow.json'
    conf.write_text(json.dumps(decl))
    e.main(['--config', str(conf), '--runs', str(tmp_path / 'runs'), '--output', str(tmp_path / 'out')])
    done = json.loads((tmp_path / 'out' / 'DONE.json').read_text())
    prov = done['provenance']
    assert prov['source_revision'] and 'endnow_data.py' in prov['scripts_sha256'] and 'encoding.py' in prov['source_files']
    from train_value_v2 import shards
    data = shards(tmp_path / 'out', 'train')
    assert len(data['y']) == (row is not None)
