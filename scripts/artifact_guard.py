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
import sys
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


def producer_provenance():
    """Producer identity for a sealed artifact (Codex HOLD @e19e5a1): source revision (a snapshot's COMMIT, else
    git HEAD plus dirty paths), every catan_ai source and engine-binary hash (``source_files``), the hash of
    every loaded module under scripts/, and the interpreter / library versions."""
    import platform
    import subprocess
    from catan_ai.experiments import source_files
    from catan_ai.engine import BACKEND
    root = Path(__file__).resolve().parents[1]
    commit = root / 'COMMIT'
    revision, dirty = (commit.read_text().strip(), []) if commit.exists() else (None, None)
    if revision is None:
        try:
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                                      timeout=10).stdout.strip() or None
            dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'experiments'], cwd=root,
                                   capture_output=True, text=True, timeout=10).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            pass
    here = Path(__file__).resolve().parent
    scripts = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in list(sys.modules.values())
               if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == here}
    versions = {'python': platform.python_version()}
    for lib in ('numpy', 'torch'):
        if lib in sys.modules:
            versions[lib] = sys.modules[lib].__version__
    return {'source_revision': revision, 'dirty_paths': dirty, 'engine_backend': BACKEND,
            'source_files': source_files(), 'scripts_sha256': dict(sorted(scripts.items())), 'versions': versions}
