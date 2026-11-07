"""Artifact safety for training and gating (Codex HOLD on PR #28, P2).

- ``refuse_existing``: every destination must be absent before any work starts.
- ``publish``: write through a temporary file and link it into place, so an
  existing artifact is never replaced.
- ``provenance`` / ``validate``: a checkpoint records the declaration it was
  trained under (config, data identity, splits, point); a gate refuses a model
  whose provenance does not match the declaration it is asked to evaluate, or
  whose training games overlap the test games.
"""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuse_existing(*paths):
    present = [str(p) for p in paths if Path(p).exists()]
    if present:
        raise FileExistsError(f'refusing to overwrite existing artifacts: {present}')


def publish(path, write):
    """``write(tmp_path)`` creates the file; it is then linked to ``path`` without clobbering."""
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    write(tmp)
    try:
        os.link(tmp, path)  # fails if path exists
    finally:
        tmp.unlink()


def provenance(cfg, data_cfg, manifest, point=None):
    return {'config': cfg, 'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            'data_identity_sha256': manifest['identity_sha256'], 'train_split': data_cfg['split']['train'],
            'test_split': data_cfg['split']['test'], 'point': point}


def validate(saved, cfg, data_cfg, manifest, point=None):
    expected = provenance(cfg, data_cfg, manifest, point)
    for key in ('config', 'data_identity_sha256', 'train_split', 'test_split', 'point'):
        if saved.get(key) != expected[key]:
            raise ValueError(f'model provenance mismatch on {key}')
    (tlo, tn), (vlo, vn) = data_cfg['split']['train'], data_cfg['split']['test']
    if set(range(tlo, tlo + tn)) & set(range(vlo, vlo + vn)):
        raise ValueError('train and test splits overlap')
