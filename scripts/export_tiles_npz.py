"""Export the arch-bakeoff ``tiles`` checkpoint (torch .pt) for NumPy serving.

Writes ``--npz`` (weights ``w:<state key>``, ``mean``, ``sd``, the tokenizer index
``tidx``, token ``types`` and a torch parity probe ``parity_x`` / ``parity_logit``) and
``--meta`` (.pt and .npz SHA-256, array shapes, the training config, source run and
the NumPy-vs-torch parity on the probe). The npz is written with fixed zip timestamps,
so the same .pt always exports to the same bytes. Weights stay local (gitignored);
only the meta file is committed.

No clobber (Codex HOLD @1d18b83, P1): both destinations must be absent before any
work starts (``artifact_guard.refuse_existing``); the pair is written and validated
(``TilesValueNet``: SHA-256, tokenizer index and NumPy-vs-torch parity) in a private
staging directory next to the destination, and only a validated pair is published
with ``artifact_guard.publish`` (hard link, never replacing an existing file). A
refusal or a failed validation leaves any existing files untouched.

    uv run --with numpy --with torch python scripts/export_tiles_npz.py \
        --pt checkpoints/value_tiles_v1.pt --npz checkpoints/value_tiles_v1.npz \
        --meta checkpoints/value_tiles_v1.meta.json

A checkpoint derived by ``scripts/tiles_margin.py win`` (trunk + win head of a tiles_margin
model, margin head dropped) is an ordinary ``tiles`` checkpoint plus ``derived_from``;
the meta then carries ``derived_from`` and ``--source-run`` names its run.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from artifact_guard import publish, refuse_existing  # noqa: E402

SOURCE_RUN = 'runs/arch-bakeoff-v1 (ultra)'
PROBE_SEED, PROBE_RANDOM, PROBE_GAME_SEED = 20261108, 64, 2690000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_torch_net(torch, pt):
    import arch_bakeoff as ab
    ckpt = torch.load(pt, map_location='cpu', weights_only=False)
    if ckpt.get('arm') != 'tiles' or ckpt.get('smoke'):
        raise ValueError('not a full (non-smoke) tiles checkpoint')
    if ckpt.get('size'):  # value-arms T7: the NumPy server is fixed at the served dimensions
        raise ValueError(f'non-default tiles size {ckpt["size"]} cannot be served by tiles_value.TilesValueNet')
    phase = 'phase_vec' in ckpt['state']  # value-arms S1: served afterstates are phase 0, so the vector is unused
    net = ab.make_net(torch, 'tiles', ab.layout(), phase=phase)
    net.load_state_dict(ckpt['state'])
    net.eval()
    return net, ckpt


def torch_logits(torch, net, ckpt, rows):
    """Exactly ``arch_bakeoff.score``'s path: float32 rows, (x - mean) / sd, net(x)."""
    with torch.no_grad():
        x = (torch.from_numpy(np.asarray(rows, dtype=np.float32)) - ckpt['mean']) / ckpt['sd']
        return net(x).double().numpy()


def probe_rows(mean, sd):
    """Fixed probe: rows whose standardized values are N(0, 1), plus real encoded positions.

    The real positions are encoded from the full game state, so they contain opponents' true hands.
    That is harmless here (the rows only check NumPy-vs-torch parity of the exported weights), but this
    pattern must never be copied into a data or serving path: those encode from the chooser's
    card-counted world (red team 2026-11-09)."""
    from catan_ai.bots import make_policy
    from catan_ai.encoding import encode
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    rng = np.random.default_rng(PROBE_SEED)
    rows = list(mean + sd * rng.standard_normal((PROBE_RANDOM, len(mean))))
    game = Game.new(list('abcd'), PROBE_GAME_SEED)
    bots = [make_policy('tactical', PROBE_GAME_SEED + i) for i in range(4)]
    step = 0
    while game.winner is None and game.turn < 200:
        seat = game.current
        if step % 7 == 0:
            for s in range(4):
                rows.append(encode(Observation.from_game(game, s, include_events=False), game))
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
        step += 1
    return np.asarray(rows, dtype=np.float32)


def write_npz(path, arrays):
    """``np.savez`` layout with fixed timestamps (byte-reproducible)."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f'{name}.npy', date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=Path, required=True)
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--source-run', default=SOURCE_RUN)
    args = parser.parse_args(argv)
    stem = str(args.npz)[:-len('.npz')]
    if not str(args.npz).endswith('.npz') or Path(f'{stem}.meta.json') != args.meta:
        parser.error('--meta must be <stem>.meta.json for --npz <stem>.npz (the loader reads both by stem)')
    refuse_existing(args.npz, args.meta)
    net, ckpt = load_torch_net(torch, args.pt)
    mean, sd = ckpt['mean'].numpy().astype(np.float32), ckpt['sd'].numpy().astype(np.float32)
    arrays = {'mean': mean, 'sd': sd}
    for k, v in ckpt['state'].items():
        if k == 'phase_vec':
            continue
        if k in ('tidx', 'types'):
            arrays[k] = v.numpy().astype(np.int64)
        else:
            arrays[f'w:{k}'] = v.numpy().astype(np.float32)
    probe = probe_rows(mean.astype(np.float64), sd.astype(np.float64))
    arrays['parity_x'] = probe
    arrays['parity_logit'] = torch_logits(torch, net, ckpt, probe)
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.export-tiles-', dir=args.npz.parent))
    try:
        staged_npz, staged_meta = staging / args.npz.name, staging / args.meta.name
        meta, diff = stage(torch, ckpt, arrays, probe, args.pt, staged_npz, staged_meta, args.source_run)
        refuse_existing(args.npz, args.meta)
        publish(args.npz, lambda tmp: shutil.copyfile(staged_npz, tmp))
        publish(args.meta, lambda tmp: shutil.copyfile(staged_meta, tmp))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({'npz': str(args.npz), 'npz_sha256': meta['npz_sha256'], 'pt_sha256': meta['pt_sha256'],
                      'probe_rows': len(probe), 'parity_max_abs_diff': diff}))


def validate(stem):
    """The serving loader's checks (SHA-256, tokenizer index, parity); returns the parity."""
    from catan_ai.tiles_value import TilesValueNet
    return TilesValueNet(stem).parity_max_abs_diff


def stage(torch, ckpt, arrays, probe, pt, npz, meta_path, source_run=SOURCE_RUN):
    """Write the pair at the staging paths and validate it there."""
    write_npz(npz, arrays)
    meta = {'schema': 'catan-value-tiles-v1', 'arm': 'tiles', 'pt_sha256': sha256(pt),
            'npz_sha256': sha256(npz), 'source_run': source_run,
            'arrays': {k: list(v.shape) for k, v in sorted(arrays.items())},
            'params': ckpt['params'], 'epoch': ckpt['epoch'], 'val_bce': ckpt['val_bce'],
            'epochs': ckpt.get('epochs'), 'train_identity': ckpt.get('train_identity', ckpt.get('train')),
            'train_seconds': ckpt.get('train_seconds'), 'config': ckpt['config'],
            'torch_version': torch.__version__,
            'probe': {'seed': PROBE_SEED, 'random_rows': PROBE_RANDOM, 'game_seed': PROBE_GAME_SEED,
                      'rows': len(probe)}}
    if 'derived_from' in ckpt:
        meta['derived_from'] = ckpt['derived_from']
    if 'value_arm' in ckpt:  # experiments/value-arms.json checkpoint
        meta.update(value_arm=ckpt['value_arm'], phase_vec_dropped='phase_vec' in ckpt['state'],
                    init_from_sha256=ckpt.get('init_from_sha256'),
                    producer_source_revision=(ckpt.get('provenance') or {}).get('source_revision'))
    meta_path.write_text(json.dumps(meta, indent=1) + '\n')
    diff = validate(str(npz)[:-len('.npz')])
    meta['parity_max_abs_diff'] = diff
    meta_path.write_text(json.dumps(meta, indent=1) + '\n')
    return meta, diff


if __name__ == '__main__':
    main()
