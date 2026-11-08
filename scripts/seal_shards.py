"""Seal a historical ``reencode_v2.py`` shard directory (Codex HOLD @c32a17a, P1).

Shard directories written before the producer recorded NPZ hashes carry sidecars that
hash only their source games, so ``train_value_v2.shards`` refuses them. This script
binds such a directory as it now stands: for every shard the declaration in
``DONE.json`` implies, it checks the inventory, the sidecar's split, seed block and row
count against the arrays and that every row's game id lies in its seed block, then
writes ``SEAL.json`` (``open('x')``) with each shard's NPZ SHA-256, sidecar SHA-256, row
count and row game-id range, stamped ``sealed_after_the_fact: <UTC time>``.

It never rewrites an NPZ, a sidecar, ``DONE.json`` or any receipt: the historical
sidecar bytes, and hence the historical ``identity`` that existing checkpoints record,
stay unchanged. A seal attests the arrays from the sealing time on; it cannot show they
equal what the producer wrote, which the timestamp makes visible.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_value_v2 import ARRAYS, check_shard, expected_shards, file_sha256, shard_stats  # noqa: E402


def seal(root, now=None):
    root = Path(root)
    path = root / 'SEAL.json'
    if path.exists():
        raise FileExistsError(f'{path} exists; a seal is never rewritten')
    done = json.loads((root / 'DONE.json').read_text())['config']
    entries = {}
    for split in done['split']:
        expected = expected_shards(done, split)
        found = {f.stem for f in root.glob(f'{split}-*.npz')} | {f.stem for f in root.glob(f'{split}-*.json')}
        if found != set(expected):
            raise ValueError(f'{split} shard inventory differs from the declaration: missing '
                             f'{sorted(set(expected) - found)}, unexpected {sorted(found - set(expected))}')
        for stem, seeds in sorted(expected.items()):
            text = (root / f'{stem}.json').read_text()
            meta = json.loads(text)
            if 'npz_sha256' in meta:
                raise ValueError(f'{stem} already carries its producer NPZ hash; nothing to seal')
            npz = root / f'{stem}.npz'
            with np.load(npz) as z:
                arrays = {k: z[k] for k in ARRAYS}
            stats = shard_stats(arrays)
            check_shard(stem, split, seeds, {**meta, 'row_games': stats['row_games']}, arrays)
            entries[stem] = {'npz_sha256': file_sha256(npz), 'sidecar_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             **stats}
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime('%Y-%m-%dT%H:%M:%SZ')
    record = {'sealed_after_the_fact': stamp,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'shards': entries}
    with path.open('x') as stream:
        json.dump(record, stream, indent=1)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    record = seal(args.root)
    print(json.dumps({'sealed_after_the_fact': record['sealed_after_the_fact'], 'shards': len(record['shards'])}))


if __name__ == '__main__':
    main()
