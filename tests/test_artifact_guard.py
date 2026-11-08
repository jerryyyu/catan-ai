import runpy
from pathlib import Path

import pytest

guard = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/artifact_guard.py'))
CFG = {'hidden': 64, 'seed': 0}
DATA = {'split': {'train': [100, 8], 'test': [108, 2]}}
MANIFEST = {'identity_sha256': 'abc'}


def test_existing_artifacts_are_refused_and_left_byte_identical(tmp_path):
    model, receipt = tmp_path / 'm.pt', tmp_path / 'r.json'
    model.write_bytes(b'trained')
    receipt.write_bytes(b'receipt')
    with pytest.raises(FileExistsError):
        guard['refuse_existing'](receipt, model)
    with pytest.raises(FileExistsError):
        guard['publish'](model, lambda tmp: Path(tmp).write_bytes(b'new'))
    assert model.read_bytes() == b'trained' and receipt.read_bytes() == b'receipt'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['m.pt', 'r.json']  # no temp left behind


def test_publish_writes_new_artifacts(tmp_path):
    guard['publish'](tmp_path / 'x.bin', lambda tmp: Path(tmp).write_bytes(b'ok'))
    assert (tmp_path / 'x.bin').read_bytes() == b'ok'


def test_gate_rejects_models_from_another_declaration():
    saved = guard['provenance'](CFG, DATA, MANIFEST, 'end_turn')
    guard['validate'](saved, CFG, DATA, MANIFEST, 'end_turn')
    for cfg, data, manifest, point in ((dict(CFG, hidden=128), DATA, MANIFEST, 'end_turn'),
                                       (CFG, DATA, {'identity_sha256': 'other'}, 'end_turn'),
                                       (CFG, DATA, MANIFEST, 'roll'),
                                       (CFG, {'split': {'train': [100, 9], 'test': [108, 2]}}, MANIFEST, 'end_turn')):
        with pytest.raises(ValueError):
            guard['validate'](saved, cfg, data, manifest, point)


def test_gate_rejects_overlapping_splits():
    data = {'split': {'train': [100, 9], 'test': [108, 2]}}
    with pytest.raises(ValueError, match='overlap'):
        guard['validate'](guard['provenance'](CFG, data, MANIFEST), CFG, data, MANIFEST)
