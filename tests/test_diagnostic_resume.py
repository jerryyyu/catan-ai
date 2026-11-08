"""Diagnostic roots are bound to one run identity; resume and analysis refuse mismatches
(Codex HOLD @c32a17a, P2)."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_search_diagnostic  # noqa: E402
import phase_diagnostic  # noqa: E402
from rank_data import canonical_sha  # noqa: E402
import search_diagnostic  # noqa: E402

CFG = {'max_turns': 50, 'decision_rate': 0.0, 'max_decisions_per_game': 1, 'candidates': 1, 'playouts': 2,
       'lineups': {'tac': ['tactical', 'tactical', 'tactical']}, 'seeds': {'tac': [123, 2]},
       'phases': [], 'max_per_game': {}}


def identity(cfg=CFG):
    body = {'config': cfg, 'script': 'test'}
    return {**body, 'identity_sha256': canonical_sha(body)}


@pytest.mark.parametrize('module', [search_diagnostic, phase_diagnostic])
def test_resume_refuses_record_of_another_seed(tmp_path, module):
    """The reproduction in the HOLD: 123.json holding seed 999 must not count as seed 123."""
    path = tmp_path / '123.json'
    ident = identity()['identity_sha256']
    path.write_text(json.dumps({'seed': 999, 'lineup': 'tac', 'identity': ident, 'decisions': []}))
    with pytest.raises(ValueError, match='does not match'):
        module.run_game((123, CFG, 'tac', str(path), ident))


@pytest.mark.parametrize('module', [search_diagnostic, phase_diagnostic])
def test_resume_refuses_record_of_another_identity_and_accepts_its_own(tmp_path, module):
    path = tmp_path / '123.json'
    ident = identity()['identity_sha256']
    path.write_text(json.dumps({'seed': 123, 'lineup': 'tac', 'identity': 'f' * 64, 'decisions': []}))
    with pytest.raises(ValueError, match='does not match'):
        module.run_game((123, CFG, 'tac', str(path), ident))
    path.write_text(json.dumps({'seed': 123, 'lineup': 'tac', 'identity': ident, 'decisions': []}))
    assert module.run_game((123, CFG, 'tac', str(path), ident)) == 'exists'


def _main(module, monkeypatch, cfg_path, root):
    monkeypatch.setattr(sys, 'argv', ['x', '--config', str(cfg_path), '--root', str(root)])
    module.main()


@pytest.mark.parametrize('module', [search_diagnostic, phase_diagnostic])
def test_main_refuses_changed_declaration_before_touching_root(tmp_path, monkeypatch, module):
    cfg = dict(CFG, seeds={'tac': [123, 0]})  # no games: only the root binding runs
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg))
    root = tmp_path / 'root'
    _main(module, monkeypatch, cfg_path, root)
    manifest = (root / 'manifest.json').read_bytes()
    assert json.loads(manifest)['config'] == cfg
    _main(module, monkeypatch, cfg_path, root)  # same identity resumes
    cfg_path.write_text(json.dumps(dict(cfg, playouts=99)))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='different run identity'):
        _main(module, monkeypatch, cfg_path, root)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('module', [search_diagnostic, phase_diagnostic])
def test_main_refuses_unbound_legacy_root_without_rewriting_config(tmp_path, monkeypatch, module):
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(dict(CFG, seeds={'tac': [123, 0]})))
    root = tmp_path / 'root'
    root.mkdir()
    legacy = json.dumps({'config': dict(CFG, playouts=64), 'script_sha256': '0' * 64})
    (root / 'config.json').write_text(legacy)
    with pytest.raises(ValueError, match='no manifest'):
        _main(module, monkeypatch, cfg_path, root)
    assert (root / 'config.json').read_text() == legacy
    assert not (root / 'manifest.json').exists()


def _decision(phase=None):
    d = {'turn': 3, 'points': [2, 2, 2, 2], 'served': {'type': 'end_turn'}, 'pv': {'type': 'end_turn'},
         'options': [{'kind': 'first', 'option': {'type': 'end_turn'}, 'bits': [1, 0, 1, 0]},
                     {'kind': 'first', 'option': {'type': 'build_road', 'edge': 1}, 'bits': [1, 1, 0, 1]}]}
    if phase:
        d['phase'] = phase
    return d


def _root(tmp_path, phase=None, seeds=(123, 124)):
    root = tmp_path / 'root'
    ident = identity()
    search_diagnostic.prepare_root(root, ident)
    (root / 'tac').mkdir()
    for seed in seeds:
        (root / 'tac' / f'{seed}.json').write_text(json.dumps(
            {'seed': seed, 'lineup': 'tac', 'identity': ident['identity_sha256'], 'decisions': [_decision(phase)]}))
    return root


def _analyze(monkeypatch, root, out):
    monkeypatch.setattr(sys, 'argv', ['x', '--root', str(root), '--output', str(out)])
    analyze_search_diagnostic.main()


def test_analysis_refuses_partial_coverage(tmp_path, monkeypatch):
    root = _root(tmp_path, seeds=(123,))
    with pytest.raises(ValueError, match='coverage'):
        _analyze(monkeypatch, root, tmp_path / 'out.json')
    with pytest.raises(ValueError, match='coverage'):
        phase_diagnostic.analyze_root(root, tmp_path / 'phase.json')
    assert not (tmp_path / 'out.json').exists()


def test_analysis_refuses_mismatched_record_and_undeclared_seed(tmp_path, monkeypatch):
    root = _root(tmp_path)
    (root / 'tac' / '124.json').write_text(json.dumps({'seed': 124, 'lineup': 'tac', 'identity': 'e' * 64,
                                                       'decisions': [_decision()]}))
    with pytest.raises(ValueError, match='does not match'):
        _analyze(monkeypatch, root, tmp_path / 'out.json')
    root2 = _root(tmp_path / 'b', seeds=(123, 124, 125))
    with pytest.raises(ValueError, match='coverage'):
        _analyze(monkeypatch, root2, tmp_path / 'out2.json')


def test_analysis_of_complete_root_is_unchanged_math(tmp_path, monkeypatch):
    import random
    root = _root(tmp_path)
    _analyze(monkeypatch, root, tmp_path / 'out.json')
    got = json.loads((tmp_path / 'out.json').read_text())
    expected = analyze_search_diagnostic.analyze([(123, _decision()), (124, _decision())], random.Random(29))
    assert got['tac'] == json.loads(json.dumps(expected))
    assert got['_identity']['legacy_unbound'] is False
    proot = _root(tmp_path / 'p', phase='robber')
    phase_diagnostic.analyze_root(proot, tmp_path / 'phase.json')
    assert json.loads((tmp_path / 'phase.json').read_text())['tac']['robber']['decisions'] == 2


def test_legacy_root_needs_explicit_flag(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    (root / 'tac').mkdir(parents=True)
    (root / 'config.json').write_text(json.dumps({'config': CFG}))
    for seed in (123, 124):
        (root / 'tac' / f'{seed}.json').write_text(json.dumps({'seed': seed, 'lineup': 'tac', 'decisions': [_decision()]}))
    with pytest.raises(ValueError, match='no manifest'):
        _analyze(monkeypatch, root, tmp_path / 'out.json')
    monkeypatch.setattr(sys, 'argv', ['x', '--root', str(root), '--output', str(tmp_path / 'out.json'), '--legacy'])
    analyze_search_diagnostic.main()
    assert json.loads((tmp_path / 'out.json').read_text())['_identity']['legacy_unbound'] is True
