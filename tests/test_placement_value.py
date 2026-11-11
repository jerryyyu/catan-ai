import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
DECL = json.loads((ROOT / 'experiments' / 'placement-value.json').read_text())


def test_declarations_are_consistent():
    for root in DECL['roots'].values():
        assert (ROOT / 'experiments' / root['declaration']).exists()
    assert set(DECL['arms']['p1']['data']) == set(DECL['roots'])
    va = json.loads((ROOT / 'experiments' / 'value-arms.json').read_text())
    assert DECL['common'] == va['common'] and DECL['served_reference']['pt_sha256'] == va['reference']['pt_sha256']
    for name, src in (('setup-v2-100k', 'value-v2-100k'), ('setup-v2-tiles', 'value-v2-tiles')):
        a, b = (json.loads((ROOT / 'experiments' / f'{n}.json').read_text()) for n in (name, src))
        assert a['roots'] == b['roots'] and a['split'] == b['split'] and a['point'] == 'setup'


def test_setupfork_is_deterministic_and_labelled():
    pytest.importorskip('numpy')
    if not (ROOT / 'checkpoints' / 'value_tiles_v1.npz').exists():
        pytest.skip('served weights absent')
    from catan_ai.engine import BACKEND
    if BACKEND != 'python':
        pytest.skip('the served binding is for the pure-Python engine')
    import setup_data as s
    a, b = s.setupfork_row(3200000, 'planner_pv2_tiles', 1000), s.setupfork_row(3200000, 'planner_pv2_tiles', 1000)
    assert a[0] is not None and a[0][1] in (0, 1) and a[0][1:] == b[0][1:] and list(a[0][0]) == list(b[0][0])


def test_served_ranking_top_is_the_served_choice():
    pytest.importorskip('numpy')
    from catan_ai.engine import BACKEND
    if BACKEND != 'python':
        pytest.skip('the served binding is for the pure-Python engine')
    import setup_data as s
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    game = Game.new(list('abcd'), 3300001)
    bot = make_policy('planner_offer_accept', 3300001 * 17 + 1)
    view = Observation.for_policy(game, 0, bot)
    assert s.served_ranking(bot, view, 0)[0] == bot.choose(view, 0)


def test_placement_metric_and_bound():
    pytest.importorskip('numpy')
    import placement_value_gate as g
    rows = lambda rates, served: [{'wins': int(r * 8), 'playouts': 8, 'served': i == served} for i, r in enumerate(rates)]
    decisions = {(s, i): rows([.25, .5, .125], 0) for s in range(20) for i in range(2)}
    better = {k: [0, 1, 0] for k in decisions}   # always picks the .5 candidate
    same = {k: [1, 0, 0] for k in decisions}     # always the served choice
    m = g.placement_metric(decisions, better)
    assert m['pick_minus_served'] == pytest.approx(.25) and m['lcb'] == pytest.approx(.25) and m['top1_agrees_with_served'] == 0
    assert g.placement_metric(decisions, same)['pick_minus_served'] == 0
    assert m['oracle_minus_served'] == pytest.approx(.25)


def test_placetest_root_must_match_its_declaration(tmp_path):
    import hashlib
    import placement_value_gate as g
    decl = json.loads((ROOT / 'experiments' / 'placement-test-served.json').read_text())
    (tmp_path / 'rows').mkdir()
    shards = {}
    lo, n = decl['split']['test']
    for seed in range(lo, lo + n):
        for i in range(8):
            for c in range(decl['top_k'] + decl['random_k']):
                f = tmp_path / 'rows' / f'{seed}-{i}-{c}.json'
                f.write_text(json.dumps({'seed': seed, 'setup_index': i, 'vertex': c, 'served': c == 0, 'x': [0.0],
                                         'wins': 1, 'playouts': 8}))
                shards[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (tmp_path / 'DONE.json').write_text(json.dumps({'config': decl, 'shards': shards}))
    out, _ = g.load_placetest(tmp_path)
    assert len(out) == 400
    (tmp_path / 'DONE.json').write_text(json.dumps({'config': dict(decl, playouts=4), 'shards': shards}))
    with pytest.raises(ValueError, match='not made under'):
        g.load_placetest(tmp_path)
    (tmp_path / 'DONE.json').write_text(json.dumps({'config': decl, 'shards': dict(shards, extra='x')}))
    with pytest.raises(ValueError, match='differ from the sealed'):
        g.load_placetest(tmp_path)


def test_setupfork_producer_seals_shards_and_provenance(tmp_path):
    pytest.importorskip('numpy')
    if not (ROOT / 'checkpoints' / 'value_tiles_v1.npz').exists():
        pytest.skip('served weights absent')
    from catan_ai.engine import BACKEND
    if BACKEND != 'python':
        pytest.skip('the served binding is for the pure-Python engine')
    import setup_data as s
    from train_value_v2 import shards
    decl = {'roots': [], 'point': 'setupfork', 'policy': 'planner_pv2_tiles', 'stride': 1, 'max_turns': 1000,
            'shard_games': 1, 'split': {'train': [3200000, 1]}}
    conf = tmp_path / 'c.json'
    conf.write_text(json.dumps(decl))
    s.main(['--config', str(conf), '--runs', str(tmp_path), '--output', str(tmp_path / 'out')])
    done = json.loads((tmp_path / 'out' / 'DONE.json').read_text())
    assert done['provenance']['source_revision'] and 'setup_data.py' in done['provenance']['scripts_sha256']
    assert len(shards(tmp_path / 'out', 'train')['y']) == 1
