"""Train one value-arms arm (experiments/value-arms.json; board #1 T7/T8/T9 and S1/S2).

Every arm is the served tile-token transformer recipe (arch_bakeoff ``tiles``: AdamW, cosine with warmup,
batch 4096, one of the 12 hex symmetries per row per batch, epoch chosen on held-out train games) changed only
as its declaration says:

* ``data``: the declared shard roots it trains on (each bound to its data declaration; see ``check_roots``);
* ``size``: tiles {d, heads, mlp, layers, dropout} (T7);
* ``init_from: reference`` warm-starts from the served checkpoint and keeps its standardization (T8);
* ``phase``: per-root 0/1 flag added to the global token through a learned vector (S1); served rows are phase 0;
* ``weight``: per-root loss weight (S2).

Validation rows are each root's last ``validation_games`` train games; the epoch is chosen on the validation
rows of the ``selection`` roots only (end-of-turn rows of the original and on-policy data).

    value_arms.py --config experiments/value-arms.json --arm t9 --runs runs --out runs/value-arms-v1 --device mps
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_bakeoff as ab  # noqa: E402
from artifact_guard import producer_provenance as provenance, publish, refuse_existing  # noqa: E402
from train_value_v2 import DATA_KEYS, make_optimizer, shards, symmetry_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / 'experiments' / 'value-arms.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_identity(path, split='train'):
    """What a checkpoint records for one training root: shard identity, array digest, the sealing DONE.json."""
    d = shards(path, split)
    return {'identity': d['identity'], 'arrays_sha256': d['arrays_sha256'], 'done_sha256': sha256(Path(path) / 'DONE.json')}


def check_training(cfg, arm, saved, runs):
    """The checkpoint must record exactly its arm's declared roots, each matching the shards on disk now."""
    _, spec = arm_spec(cfg, arm)
    recorded = saved.get('train') or {}
    if sorted(recorded) != sorted(spec['data']):
        raise ValueError(f'{arm}: checkpoint records roots {sorted(recorded)}, declared {sorted(spec["data"])}')
    for name, (path, _) in check_roots(cfg, runs, spec['data']).items():
        now = root_identity(path)
        for key in ('identity', 'arrays_sha256', 'done_sha256'):
            if recorded[name].get(key) != now[key]:
                raise ValueError(f'{arm}: {name} {key} differs from the shards on disk')


def arm_spec(cfg, name):
    if name not in cfg['arms']:
        raise ValueError(f'arm {name} is not declared')
    spec = cfg['arms'][name]
    unknown = set(spec) - {'purpose', 'data', 'size', 'init_from', 'phase', 'weight', 'common'}
    if unknown:
        raise ValueError(f'{name}: undeclared keys {sorted(unknown)}')
    if any(r not in cfg['roots'] for r in spec['data']):
        raise ValueError(f'{name}: data root not declared')
    return {**cfg['common'], **spec.get('common', {})}, spec


def check_roots(cfg, runs, names):
    """Each root's shards must be completed under its declaration's data keys (Codex HOLD @3b390e8 pattern)."""
    out = {}
    for name in names:
        root = cfg['roots'][name]
        decl = json.loads((ROOT / 'experiments' / root['declaration']).read_text())
        done = Path(runs) / root['shards'] / 'DONE.json'
        if not done.exists():
            raise ValueError(f'{name}: {done} missing (not a completed shard set)')
        made = json.loads(done.read_text())['config']
        differ = [k for k in DATA_KEYS if made.get(k) != decl.get(k)]
        if differ:
            raise ValueError(f'{name}: shards not made under {root["declaration"]} (differs on {differ})')
        out[name] = (Path(runs) / root['shards'], decl['split'])
    return out


def load(cfg, spec, roots, max_rows, seed):
    """Concatenated train rows with per-row phase, weight, validation and selection masks."""
    parts, ident = [], {}
    for name in spec['data']:
        path, split = roots[name]
        d = ab.subsample(shards(path, 'train'), max_rows, seed)
        lo, n = split['train']
        if d['game'].min() < lo or d['game'].max() >= lo + n:
            raise ValueError(f'{name}: train rows outside the declared split')
        val = d['game'] >= lo + n - cfg['common']['validation_games']
        k = len(d['y'])
        parts.append((d['x'], d['y'], np.full(k, spec.get('phase', {}).get(name, 0), np.float32),
                      np.full(k, spec.get('weight', {}).get(name, 1.0), np.float32), val,
                      np.full(k, name in cfg['selection'], bool)))
        ident[name] = {'identity': d['identity'], 'arrays_sha256': d['arrays_sha256'],
                       'done_sha256': sha256(Path(path) / 'DONE.json')}
    cat = [np.concatenate([p[i] for p in parts]) for i in range(6)]
    return cat, ident


def train(args, cfg):
    import torch
    common, spec = arm_spec(cfg, args.arm)
    epochs = args.epochs or common['epochs']
    torch.manual_seed(common['seed'])
    device = ab.device_of(torch, args.device)
    board = ab.layout()
    roots = check_roots(cfg, args.runs, spec['data'])
    (x, y, phase, weight, val, select), ident = load(cfg, spec, roots, args.max_rows, common['seed'])
    fit_rows, val_rows = np.flatnonzero(~val), np.flatnonzero(val & select)
    if not len(fit_rows) or not len(val_rows):
        raise ValueError('empty fit or selection-validation rows')
    perms_cpu = symmetry_index(torch)
    use_phase = bool(spec.get('phase'))
    net = ab.make_net(torch, 'tiles', board, size=spec.get('size'), phase=use_phase)
    reference = None
    if spec.get('init_from') == 'reference':
        ref_path = Path(args.runs) / cfg['reference']['pt']
        if sha256(ref_path) != cfg['reference']['pt_sha256']:
            raise ValueError('reference checkpoint differs from the declared served checkpoint')
        reference = torch.load(ref_path, weights_only=False, map_location='cpu')
        net.load_state_dict(reference['state'])
        mean, sd = reference['mean'], reference['sd']
    else:
        mean, sd = ab.standardization(torch, x, fit_rows, perms_cpu)
    net = net.to(device)
    perms, dmean, dsd = perms_cpu.to(device), mean.to(device), sd.to(device)
    params = sum(p.numel() for p in net.parameters())
    batch = common['batch']
    steps = epochs * ((len(fit_rows) + batch - 1) // batch)
    opt, sched = make_optimizer(torch, net, common, steps)
    gen = torch.Generator().manual_seed(common['seed'])
    yt, pt_, wt = (torch.from_numpy(a.astype(np.float32)) for a in (y, phase, weight))
    yv = y[val_rows].astype(np.float64)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    print(json.dumps({'arm': args.arm, 'device': str(device), 'params': params, 'fit_rows': len(fit_rows),
                      'selection_val_rows': len(val_rows), 'epochs': epochs, 'roots': spec['data']}), flush=True)
    best, history, started = None, [], time.monotonic()
    for epoch in range(epochs):
        net.train()
        loss_sum = w_sum = 0.
        for idx in torch.randperm(len(fit_rows), generator=gen).split(batch):
            rows = np.sort(fit_rows[idx.numpy()])
            xb = ab.to_batch(torch, x, rows, dmean, dsd, device)
            pick = torch.randint(0, len(perms_cpu), (len(rows),), generator=gen).to(device)
            xb = torch.gather(xb, 1, perms[pick])
            rt = torch.from_numpy(rows)
            out = net(xb, pt_[rt].to(device)) if use_phase else net(xb)
            w = wt[rt].to(device)
            loss = (bce(out, yt[rt].to(device), reduction='none') * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            loss_sum += loss.item() * float(w.sum())
            w_sum += float(w.sum())
        p, _ = ab.score(torch, net, 'tiles', x, val_rows, dmean, dsd, device)  # phase 0: selection rows are end-of-turn
        entry = {'epoch': epoch, 'train_loss': loss_sum / w_sum, 'val_bce': ab.log_loss(p, yv),
                 'val_auc': ab.fast_auc(p, yv), 'seconds': time.monotonic() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if best is None or entry['val_bce'] < best[0]:
            best = (entry['val_bce'], epoch, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    smoke = bool(args.max_rows or args.epochs)
    path = args.out / f'{args.arm}.pt'
    publish(path, lambda tmp: torch.save({
        'arm': 'tiles', 'value_arm': args.arm, 'state': best[2], 'mean': mean, 'sd': sd, 'size': spec.get('size'),
        'epoch': best[1], 'val_bce': best[0], 'config': cfg, 'train': ident, 'params': params, 'smoke': smoke,
        'init_from_sha256': cfg['reference']['pt_sha256'] if reference is not None else None,
        'provenance': provenance()}, tmp))
    return {'arm': args.arm, 'fit_rows': len(fit_rows), 'selection_val_rows': len(val_rows), 'chosen_epoch': best[1],
            'chosen_val_bce': best[0], 'history': history, 'params': params, 'device': str(device), 'smoke': smoke,
            'train': ident, 'model_sha256': sha256(path), 'seconds': time.monotonic() - started,
            'provenance': provenance()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DECL)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only (per root)')
    parser.add_argument('--epochs', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f'{args.arm}-train.json'
    refuse_existing(output, args.out / f'{args.arm}.pt')
    result = train(args, cfg)
    result.update({'config_sha256': sha256(args.config), 'script_sha256': sha256(__file__)})
    publish(output, lambda tmp: tmp.write_text(json.dumps(result, indent=1)))
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=1))
    return result


if __name__ == '__main__':
    main()
