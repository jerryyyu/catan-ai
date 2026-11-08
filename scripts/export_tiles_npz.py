"""Export the arch-bakeoff ``tiles`` checkpoint (torch .pt) for NumPy serving.

Writes ``--npz`` (weights ``w:<state key>``, ``mean``, ``sd``, the tokenizer index
``tidx``, token ``types`` and a torch parity probe ``parity_x`` / ``parity_logit``) and
``--meta`` (.pt and .npz SHA-256, array shapes, the training config, source run and
the NumPy-vs-torch parity on the probe). The npz is written with fixed zip timestamps,
so the same .pt always exports to the same bytes. Weights stay local (gitignored);
only the meta file is committed.

    uv run --with numpy --with torch python scripts/export_tiles_npz.py \
        --pt checkpoints/value_tiles_v1.pt --npz checkpoints/value_tiles_v1.npz \
        --meta checkpoints/value_tiles_v1.meta.json
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

SOURCE_RUN = 'runs/arch-bakeoff-v1 (ultra)'
PROBE_SEED, PROBE_RANDOM, PROBE_GAME_SEED = 20261108, 64, 2690000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_torch_net(torch, pt):
    import arch_bakeoff as ab
    ckpt = torch.load(pt, map_location='cpu', weights_only=False)
    if ckpt.get('arm') != 'tiles' or ckpt.get('smoke'):
        raise ValueError('not a full (non-smoke) tiles checkpoint')
    net = ab.make_net(torch, 'tiles', ab.layout())
    net.load_state_dict(ckpt['state'])
    net.eval()
    return net, ckpt


def torch_logits(torch, net, ckpt, rows):
    """Exactly ``arch_bakeoff.score``'s path: float32 rows, (x - mean) / sd, net(x)."""
    with torch.no_grad():
        x = (torch.from_numpy(np.asarray(rows, dtype=np.float32)) - ckpt['mean']) / ckpt['sd']
        return net(x).double().numpy()


def probe_rows(mean, sd):
    """Fixed probe: rows whose standardized values are N(0, 1), plus real encoded positions."""
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
    args = parser.parse_args(argv)
    stem = str(args.npz)[:-len('.npz')]
    if not str(args.npz).endswith('.npz') or Path(f'{stem}.meta.json') != args.meta:
        parser.error('--meta must be <stem>.meta.json for --npz <stem>.npz (the loader reads both by stem)')
    net, ckpt = load_torch_net(torch, args.pt)
    mean, sd = ckpt['mean'].numpy().astype(np.float32), ckpt['sd'].numpy().astype(np.float32)
    arrays = {'mean': mean, 'sd': sd}
    for k, v in ckpt['state'].items():
        if k in ('tidx', 'types'):
            arrays[k] = v.numpy().astype(np.int64)
        else:
            arrays[f'w:{k}'] = v.numpy().astype(np.float32)
    probe = probe_rows(mean.astype(np.float64), sd.astype(np.float64))
    arrays['parity_x'] = probe
    arrays['parity_logit'] = torch_logits(torch, net, ckpt, probe)
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    write_npz(args.npz, arrays)
    meta = {'schema': 'catan-value-tiles-v1', 'arm': 'tiles', 'pt_sha256': sha256(args.pt),
            'npz_sha256': sha256(args.npz), 'source_run': SOURCE_RUN,
            'arrays': {k: list(v.shape) for k, v in sorted(arrays.items())},
            'params': ckpt['params'], 'epoch': ckpt['epoch'], 'val_bce': ckpt['val_bce'],
            'epochs': ckpt['epochs'], 'train_identity': ckpt['train_identity'],
            'train_seconds': ckpt['train_seconds'], 'config': ckpt['config'],
            'torch_version': torch.__version__,
            'probe': {'seed': PROBE_SEED, 'random_rows': PROBE_RANDOM, 'game_seed': PROBE_GAME_SEED,
                      'rows': len(probe)}}
    args.meta.write_text(json.dumps(meta, indent=1) + '\n')
    from catan_ai.tiles_value import TilesValueNet  # verifies sha, tokenizer index and parity
    served = TilesValueNet(stem)
    meta['parity_max_abs_diff'] = served.parity_max_abs_diff
    args.meta.write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({'npz': str(args.npz), 'npz_sha256': meta['npz_sha256'], 'pt_sha256': meta['pt_sha256'],
                      'probe_rows': len(probe), 'parity_max_abs_diff': served.parity_max_abs_diff}))


if __name__ == '__main__':
    main()
