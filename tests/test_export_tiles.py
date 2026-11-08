"""export_tiles_npz.py never replaces an existing artifact (Codex HOLD @1d18b83, P1)."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

pytest.importorskip('numpy')
pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import export_tiles_npz as ex  # noqa: E402

PT = ROOT / 'checkpoints' / 'value_tiles_v1.pt'
NPZ_SHA256 = '27dd7818cdc29082dd5c712c0f1fdf708ff568cdc98fb4b898f01e3f60820697'
needs_pt = pytest.mark.skipif(not PT.exists(), reason='local tiles weights (gitignored) not present')


def args(d, stem='value_tiles_v1', pt=PT):
    return ['--pt', str(pt), '--npz', str(d / f'{stem}.npz'), '--meta', str(d / f'{stem}.meta.json')]


def existing(d, stem='value_tiles_v1'):
    d.mkdir(exist_ok=True)
    (d / f'{stem}.npz').write_bytes(b'prior weights')
    (d / f'{stem}.meta.json').write_bytes(b'{"prior": "metadata"}\n')
    return state(d)


def state(d):
    """Every entry of ``d`` (a leftover staging directory shows up as 'dir')."""
    return {p.name: p.read_bytes() if p.is_file() else 'dir' for p in d.iterdir()}


@pytest.mark.parametrize('which', ['both', 'npz', 'meta'])
def test_refuses_any_existing_destination_before_work(tmp_path, which):
    d = tmp_path / 'out'
    before = existing(d)
    if which != 'both':
        (d / f"value_tiles_v1.{'meta.json' if which == 'npz' else 'npz'}").unlink()
        before = state(d)
    # a missing checkpoint proves the refusal happens before the .pt is even read
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        ex.main(args(d, pt=tmp_path / 'absent.pt'))
    assert state(d) == before


@needs_pt
def test_failed_parity_leaves_existing_files_and_publishes_nothing(tmp_path, monkeypatch):
    d = tmp_path / 'out'
    before = existing(d)  # a prior export in the same directory

    def fail(stem):
        assert Path(f'{stem}.npz').exists() and Path(stem).parent != d  # validated in private staging
        raise ValueError('numpy forward pass disagrees with the exported torch logits (forced)')
    monkeypatch.setattr(ex, 'validate', fail)
    with pytest.raises(ValueError, match='forced'):
        ex.main(args(d, stem='value_tiles_v2'))
    assert state(d) == before  # no new pair, no staging leftovers, prior bytes intact


@needs_pt
def test_destination_appearing_during_staging_is_not_replaced(tmp_path, monkeypatch):
    d = tmp_path / 'out'
    d.mkdir()
    real = ex.validate

    def race(stem):
        diff = real(stem)
        existing(d)
        return diff
    monkeypatch.setattr(ex, 'validate', race)
    before = None
    with pytest.raises(FileExistsError):
        before = state(d)
        ex.main(args(d))
    assert before == {} and state(d) == {'value_tiles_v1.npz': b'prior weights',
                                         'value_tiles_v1.meta.json': b'{"prior": "metadata"}\n'}


@needs_pt
def test_export_is_byte_identical_and_matches_the_committed_meta(tmp_path):
    d = tmp_path / 'out'
    d.mkdir()
    ex.main(args(d))
    assert sorted(state(d)) == ['value_tiles_v1.meta.json', 'value_tiles_v1.npz']
    assert hashlib.sha256((d / 'value_tiles_v1.npz').read_bytes()).hexdigest() == NPZ_SHA256
    meta = json.loads((d / 'value_tiles_v1.meta.json').read_text())
    assert meta['npz_sha256'] == NPZ_SHA256
    committed = ROOT / 'checkpoints' / 'value_tiles_v1.meta.json'
    assert (d / 'value_tiles_v1.meta.json').read_bytes() == committed.read_bytes()
    # and a second export to the same place refuses, leaving the first intact
    first = state(d)
    with pytest.raises(FileExistsError):
        ex.main(args(d))
    assert state(d) == first
    shutil.rmtree(d)
