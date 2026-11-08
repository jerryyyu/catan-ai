"""Shard identity is bound to the NPZ arrays and the declared split (Codex HOLD @c32a17a, P1)."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_value_v2  # noqa: E402

SPLIT = {'train': [100, 4], 'test': [200, 2]}


def write(root, stem, split, seeds, games, labels, producer_hash=True):
    """A shard as ``reencode_v2.py`` writes it (``producer_hash=False``: the historical format)."""
    games = np.asarray(games, dtype=np.int32)
    npz = root / f'{stem}.npz'
    with npz.open('wb') as stream:
        np.savez(stream, x=np.ones((len(games), 3), dtype=np.float16), y=np.asarray(labels, dtype=np.uint8),
                 game=games, leaf=np.zeros(len(games), dtype=np.float32), shared=np.zeros(len(games), dtype=np.float32))
    meta = {'split': split, 'seeds': seeds, 'rows': len(games)}
    if producer_hash:
        meta.update({'row_games': [int(games.min()), int(games.max())],
                     'npz_sha256': hashlib.sha256(npz.read_bytes()).hexdigest()})
    meta['source_sha256'] = {str(s): 'ab' * 32 for s in range(seeds[0], seeds[1] + 1)}
    (root / f'{stem}.json').write_text(json.dumps(meta))


def make(root, producer_hash=True):
    root.mkdir()
    (root / 'DONE.json').write_text(json.dumps({'config': {'split': SPLIT, 'shard_games': 2}}))
    write(root, 'train-0000', 'train', [100, 101], [100, 100, 101], [0, 1, 0], producer_hash)
    write(root, 'train-0001', 'train', [102, 103], [102, 103], [1, 0], producer_hash)
    write(root, 'test-0000', 'test', [200, 201], [200, 201, 201], [1, 0, 1], producer_hash)
    return root


def test_accepts_declared_sealed_shards(tmp_path):
    root = make(tmp_path / 'd')
    train = train_value_v2.shards(root, 'train')
    assert train['game'].tolist() == [100, 100, 101, 102, 103]
    assert train['sealed_after_the_fact'] is None
    assert train_value_v2.shards(root, 'test')['y'].tolist() == [1, 0, 1]


def test_changed_array_with_retained_sidecar_is_refused(tmp_path):
    """The reproduction in the HOLD: rewrite a label 0->1 and a game id, keep the sidecar."""
    root = make(tmp_path / 'd')
    before = train_value_v2.shards(root, 'train')['identity']
    sidecar = (root / 'train-0000.json').read_bytes()
    write(root, 'train-0000', 'train', [100, 101], [100, 100, 9999999], [1, 1, 0])
    (root / 'train-0000.json').write_bytes(sidecar)
    with pytest.raises(ValueError, match='NPZ bytes differ'):
        train_value_v2.shards(root, 'train')
    # A consistent rewrite (sidecar re-hashed) changes the identity the checkpoint binds.
    write(root, 'train-0000', 'train', [100, 101], [100, 100, 101], [1, 1, 0])
    after = train_value_v2.shards(root, 'train')
    assert after['identity'] != before


def test_missing_shard_is_refused(tmp_path):
    root = make(tmp_path / 'd')
    (root / 'train-0001.npz').unlink()
    (root / 'train-0001.json').unlink()
    with pytest.raises(ValueError, match='inventory'):
        train_value_v2.shards(root, 'train')


def test_unexpected_shard_is_refused(tmp_path):
    root = make(tmp_path / 'd')
    write(root, 'train-0002', 'train', [104, 105], [104], [0])
    with pytest.raises(ValueError, match='inventory'):
        train_value_v2.shards(root, 'train')


def test_wrong_split_rows_are_refused(tmp_path):
    """A test shard whose rows come from train games (overlap) is refused."""
    root = make(tmp_path / 'd')
    write(root, 'test-0000', 'test', [200, 201], [200, 101], [1, 0])
    with pytest.raises(ValueError, match='outside its test seed block'):
        train_value_v2.shards(root, 'test')


def test_sidecar_claiming_another_block_is_refused(tmp_path):
    root = make(tmp_path / 'd')
    write(root, 'test-0000', 'train', [102, 103], [102, 103], [1, 0])
    with pytest.raises(ValueError, match='sidecar claims'):
        train_value_v2.shards(root, 'test')


def test_row_count_mismatch_is_refused(tmp_path):
    root = make(tmp_path / 'd')
    meta = json.loads((root / 'train-0001.json').read_text())
    meta['rows'] = 7
    (root / 'train-0001.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='row count'):
        train_value_v2.shards(root, 'train')


def test_historical_unsealed_refused_then_sealed_after_the_fact(tmp_path):
    import seal_shards
    root = make(tmp_path / 'd', producer_hash=False)
    with pytest.raises(ValueError, match='unsealed'):
        train_value_v2.shards(root, 'train')
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    legacy = hashlib.sha256(''.join((root / f'train-000{i}.json').read_text() for i in (0, 1)).encode()).hexdigest()
    record = seal_shards.seal(root)
    assert record['sealed_after_the_fact'].endswith('Z')
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.name != 'SEAL.json'} == before
    train = train_value_v2.shards(root, 'train')
    assert train['identity'] == legacy  # historical checkpoints' identity is not relabelled
    assert train['sealed_after_the_fact'] == record['sealed_after_the_fact']
    with pytest.raises(FileExistsError):
        seal_shards.seal(root)
    write(root, 'train-0001', 'train', [102, 103], [102, 103], [0, 0], producer_hash=False)
    with pytest.raises(ValueError, match='NPZ bytes differ'):
        train_value_v2.shards(root, 'train')


def test_seal_refuses_inconsistent_historical_shards(tmp_path):
    import seal_shards
    root = make(tmp_path / 'd', producer_hash=False)
    write(root, 'test-0000', 'test', [200, 201], [200, 103], [1, 0], producer_hash=False)
    with pytest.raises(ValueError, match='outside'):
        seal_shards.seal(root)
    assert not (root / 'SEAL.json').exists()


def test_producer_write_shard_is_accepted(tmp_path):
    import reencode_v2
    root = tmp_path / 'd'
    root.mkdir()
    (root / 'DONE.json').write_text(json.dumps({'config': {'split': {'train': [10, 2], 'test': [20, 1]},
                                                           'shard_games': 2}}))
    results = [(10, [[0.] * 3, [1.] * 3], [0, 1], [(1, 0, .5, .2), (2, 0, .4, .1)], 'cd' * 32),
               (11, [[2.] * 3], [1], [(1, 1, .3, .3)], 'ef' * 32)]
    reencode_v2.write_shard(root, 'train', 0, [10, 11], results)
    reencode_v2.write_shard(root, 'test', 0, [20], [(20, [[1.] * 3], [0], [(1, 0, .1, .1)], '01' * 32)])
    meta = json.loads((root / 'train-0000.json').read_text())
    assert meta['rows'] == 3 and meta['row_games'] == [10, 11]
    assert train_value_v2.shards(root, 'train')['y'].tolist() == [0, 1, 1]
    with pytest.raises(FileExistsError):
        reencode_v2.write_shard(root, 'test', 0, [20], [(20, [[1.] * 3], [0], [(1, 0, .1, .1)], '01' * 32)])
