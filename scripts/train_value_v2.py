"""Train and gate a state-value MLP on re-encoded (encoder v2) value shards.

Input: the shard directory written by ``reencode_v2.py`` for the declaration in
``--config`` (e.g. ``experiments/value-v2-100k.json``). The model is an MLP on the
1,324 encoder-v2 features with BCE to "this seat won". The epoch is chosen on the
last ``validation_games`` train games. ``gate`` reads the test shards once and
reports pooled AUC for the network, the served leaf and the shared score, with a
game-clustered bootstrap of network AUC minus the better baseline. Provenance and
no-clobber rules from ``artifact_guard``; the data identity is the hash of the
shard manifests (which carry their source games' hashes).

Shard identity is bound to the arrays (Codex HOLD @c32a17a, P1): ``shards`` refuses a
directory unless the shard inventory is exactly what ``DONE.json``'s declaration
implies (no missing or extra shard), every sidecar matches its shard's declared split
and seed block, every NPZ's bytes match their recorded SHA-256 (written by
``reencode_v2.py`` in the sidecar, or for historical directories in ``SEAL.json`` by
``seal_shards.py``, marked ``sealed_after_the_fact``), row counts agree, and every row's
game id lies inside its shard's seed block (hence inside its split; the declared splits
must be disjoint). ``identity`` keeps its historical formula (hash of the sidecar
texts; for producer-sealed shards this already covers the NPZ hashes);
``arrays_sha256`` hashes the verified NPZ digests and is bound into new checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_guard import publish, refuse_existing, sha256  # noqa: E402
from train_value import auc  # noqa: E402


ARRAYS = ('x', 'y', 'game', 'leaf', 'shared')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shard_stats(arrays):
    """Row count and game-id range of one shard's arrays (all arrays must have equal length)."""
    n = len(arrays['y'])
    if any(len(arrays[k]) != n for k in ARRAYS):
        raise ValueError('shard arrays have unequal lengths')
    games = arrays['game']
    return {'rows': int(n), 'row_games': [int(games.min()), int(games.max())] if n else None}


def expected_shards(done_cfg, split):
    """``{stem: [first_seed, last_seed]}`` that ``reencode_v2.py`` writes for ``split``."""
    lo, n = done_cfg['split'][split]
    size = done_cfg['shard_games']
    return {f'{split}-{start // size:04d}': [lo + start, lo + min(start + size, n) - 1] for start in range(0, n, size)}


def check_shard(stem, split, seeds, meta, arrays):
    """Refuse a shard whose sidecar or rows disagree with its declared split and seed block."""
    if meta.get('split') != split or meta.get('seeds') != seeds:
        raise ValueError(f'{stem}: sidecar claims {meta.get("split")} {meta.get("seeds")}, declared {split} {seeds}')
    stats = shard_stats(arrays)
    if meta.get('rows') != stats['rows'] or meta.get('row_games', stats['row_games']) != stats['row_games']:
        raise ValueError(f'{stem}: row count or game range differs from the sidecar')
    games = arrays['game']
    if len(games) and (int(games.min()) < seeds[0] or int(games.max()) > seeds[1]):
        raise ValueError(f'{stem}: rows from games outside its {split} seed block {seeds}')
    if set(meta.get('source_sha256', {})) != {str(s) for s in range(seeds[0], seeds[1] + 1)}:
        raise ValueError(f'{stem}: source games differ from its seed block')
    return stats


def shards(root, split):
    root = Path(root)
    done_path = root / 'DONE.json'
    if not done_path.exists():
        raise ValueError(f'{root} has no DONE.json (incomplete shard directory)')
    done = json.loads(done_path.read_text())['config']
    if split not in done['split']:
        raise ValueError(f'split {split} is not declared in {done_path}')
    blocks = [set(range(lo, lo + n)) for lo, n in done['split'].values()]
    if any(a & b for i, a in enumerate(blocks) for b in blocks[i + 1:]):
        raise ValueError('declared splits overlap')
    expected = expected_shards(done, split)
    found = {f.stem for f in root.glob(f'{split}-*.npz')} | {f.stem for f in root.glob(f'{split}-*.json')}
    if found != set(expected):
        raise ValueError(f'{split} shard inventory differs from the declaration: missing '
                         f'{sorted(set(expected) - found)}, unexpected {sorted(found - set(expected))}')
    seal_path = root / 'SEAL.json'
    seal = json.loads(seal_path.read_text()) if seal_path.exists() else {'shards': {}}
    texts, digests, data, used_seal = [], [], [], False
    for stem, seeds in sorted(expected.items()):
        text = (root / f'{stem}.json').read_text()
        meta = json.loads(text)
        claimed = meta.get('npz_sha256')
        if claimed is None:
            sealed = seal['shards'].get(stem)
            if sealed is None:
                raise ValueError(f'{stem}: no NPZ hash (unsealed historical shard; run scripts/seal_shards.py)')
            if sealed['sidecar_sha256'] != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError(f'{stem}: sidecar changed after sealing')
            if sealed['rows'] != meta.get('rows'):
                raise ValueError(f'{stem}: sealed row count differs from the sidecar')
            claimed, used_seal = sealed['npz_sha256'], True
            meta = {**meta, 'row_games': sealed['row_games']}
        npz = root / f'{stem}.npz'
        digest = file_sha256(npz)
        if digest != claimed:
            raise ValueError(f'{stem}: NPZ bytes differ from the recorded SHA-256')
        with np.load(npz) as z:
            arrays = {k: z[k] for k in ARRAYS}
        check_shard(stem, split, seeds, meta, arrays)
        texts.append(text)
        digests.append(digest)
        data.append(arrays)
    out = {k: np.concatenate([d[k] for d in data]) for k in ARRAYS}
    out['identity'] = hashlib.sha256(''.join(texts).encode()).hexdigest()
    out['arrays_sha256'] = hashlib.sha256(''.join(digests).encode()).hexdigest()
    out['sealed_after_the_fact'] = seal.get('sealed_after_the_fact') if used_seal else None
    return out


DATA_KEYS = ('roots', 'point', 'stride', 'max_turns', 'shard_games', 'split')


def build_model(torch, dim, m):
    """``mlp``: Linear/ReLU stack. ``resmlp``: input projection, residual blocks of
    LayerNorm -> Linear -> ReLU -> Dropout -> Linear (+ skip), final LayerNorm, head."""
    nn = torch.nn
    if m.get('arch', 'mlp') == 'mlp':
        layers, width = [], dim
        for _ in range(m['layers']):
            layers += [nn.Linear(width, m['hidden']), nn.ReLU()]
            if m.get('dropout'):
                layers.append(nn.Dropout(m['dropout']))
            width = m['hidden']
        return nn.Sequential(*layers, nn.Linear(width, 1))

    class Block(nn.Module):
        def __init__(self, width, dropout):
            super().__init__()
            self.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(width, width))

        def forward(self, x):
            return x + self.f(x)

    class ResMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(dim, m['hidden'])
            self.blocks = nn.Sequential(*[Block(m['hidden'], m.get('dropout', 0.)) for _ in range(m['blocks'])])
            self.out = nn.Sequential(nn.LayerNorm(m['hidden']), nn.Linear(m['hidden'], 1))

        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))
    return ResMLP()


def make_optimizer(torch, net, m, steps):
    if m.get('optimizer', 'adam') == 'adamw':
        opt = torch.optim.AdamW(net.parameters(), lr=m['learning_rate'], weight_decay=m['weight_decay'])
    else:
        opt = torch.optim.Adam(net.parameters(), lr=m['learning_rate'], weight_decay=m['weight_decay'])
    if m.get('schedule') != 'cosine':
        return opt, None
    warm = max(1, int(m.get('warmup_fraction', .03) * steps))
    import math

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return .5 * (1 + math.cos(math.pi * min(1., (step - warm) / max(1, steps - warm))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def symmetry_index(torch):
    from catan_ai.encoding import symmetry_permutations
    from catan_ai.engine import Game
    return torch.tensor(symmetry_permutations(Game.new(list('abcd'), 0).board), dtype=torch.long)


def provenance(cfg, train_identity, train_arrays):
    return {'config': cfg, 'train_identity': train_identity, 'train_arrays_sha256': train_arrays, 'split': cfg['split']}


def train(args, cfg):
    import torch
    m = cfg['model']
    torch.manual_seed(m['seed'])
    torch.set_num_threads(m['threads'])
    data = shards(args.data, 'train')
    lo, n = cfg['split']['train']
    cut = lo + n - m['validation_games']
    fit, val = data['game'] < cut, data['game'] >= cut
    x = torch.from_numpy(data['x'][fit].astype(np.float32))
    y = torch.from_numpy(data['y'][fit].astype(np.float32))
    vx = torch.from_numpy(data['x'][val].astype(np.float32))
    vy = torch.from_numpy(data['y'][val].astype(np.float32))
    mean, var = x.mean(0), x.var(0)
    perms = symmetry_index(torch) if m.get('augment') else None
    if perms is not None:
        # Symmetrize the standardization over the 12 board symmetries so it is invariant
        # under them: permuting standardized rows then equals standardizing permuted rows.
        mean = torch.stack([mean[p] for p in perms]).mean(0)
        var = torch.stack([var[p] for p in perms]).mean(0)
    sd = var.sqrt().clamp_min(1e-6)
    x, vx = (x - mean) / sd, (vx - mean) / sd
    net = build_model(torch, x.shape[1], m)
    steps = m['epochs'] * ((len(x) + m['batch'] - 1) // m['batch'])
    opt, sched = make_optimizer(torch, net, m, steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(m['seed'])
    best, history = None, []
    for epoch in range(m['epochs']):
        net.train()
        for idx in torch.randperm(len(x), generator=gen).split(m['batch']):
            xb = x[idx]
            if perms is not None:  # a random board symmetry per row (standardization is symmetric, see above)
                pick = torch.randint(0, len(perms), (len(idx),), generator=gen)
                xb = torch.gather(xb, 1, perms[pick])
            loss = lossf(net(xb).squeeze(-1), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(vx).squeeze(-1), vy).item()
        history.append({'epoch': epoch, 'val_loss': vloss})
        print(json.dumps(history[-1]), flush=True)
        if best is None or vloss < best[0]:
            best = (vloss, epoch, {k: v.clone() for k, v in net.state_dict().items()})
    publish(args.model, lambda tmp: torch.save({'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1],
                                                **provenance(cfg, data['identity'], data['arrays_sha256'])}, tmp))
    return {'fit_rows': len(x), 'val_rows': len(vx), 'best_epoch': best[1], 'history': history,
            'train_identity': data['identity'], 'train_arrays_sha256': data['arrays_sha256'],
            'train_sealed_after_the_fact': data['sealed_after_the_fact'], 'model_sha256': sha256(args.model)}


def gate(args, cfg):
    import torch
    saved = torch.load(args.model, weights_only=False)
    trained = shards(args.data, 'train')
    if saved['config'] != cfg or saved['train_identity'] != trained['identity']:
        raise ValueError('model provenance does not match this declaration and training data')
    # Checkpoints trained before array binding carry only the sidecar identity; their
    # arrays are verified against the (after-the-fact) seal by ``shards`` above.
    if saved.get('train_arrays_sha256', trained['arrays_sha256']) != trained['arrays_sha256']:
        raise ValueError('model was trained on different shard arrays')
    test = shards(args.data, 'test')
    net = build_model(torch, test['x'].shape[1], cfg['model'])
    net.load_state_dict(saved['state'])
    net.eval()
    with torch.no_grad():
        pred = net((torch.from_numpy(test['x'].astype(np.float32)) - saved['mean']) / saved['sd']).squeeze(-1).numpy()
    y = test['y'].astype(int)
    by_game = {}
    for i, g in enumerate(test['game']):
        by_game.setdefault(int(g), []).append(i)
    ids = sorted(by_game)

    def metrics(index):
        lab = [int(y[i]) for i in index]
        return {'net': auc([float(pred[i]) for i in index], lab), 'leaf': auc([float(test['leaf'][i]) for i in index], lab),
                'shared': auc([float(test['shared'][i]) for i in index], lab)}
    point = metrics(range(len(y)))
    rng = random.Random(cfg['gate']['bootstrap_seed'])
    diffs = []
    for _ in range(cfg['gate']['bootstrap_resamples']):
        sample = [i for gid in (rng.choice(ids) for _ in ids) for i in by_game[gid]]
        mm = metrics(sample)
        diffs.append(mm['net'] - max(mm['leaf'], mm['shared']))
    diffs.sort()
    lcb = diffs[int(.025 * len(diffs))]
    return {'test_rows': len(y), 'test_games': len(ids), 'auc': point,
            'gain_vs_best_baseline': point['net'] - max(point['leaf'], point['shared']), 'gain_lcb95': lcb,
            'test_identity': test['identity'], 'test_arrays_sha256': test['arrays_sha256'],
            'train_arrays_bound_in_checkpoint': 'train_arrays_sha256' in saved,
            'sealed_after_the_fact': {'train': trained['sealed_after_the_fact'], 'test': test['sealed_after_the_fact']},
            'model_sha256': sha256(args.model), 'pass': lcb > 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'gate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    done = Path(args.data) / 'DONE.json'
    made = json.loads(done.read_text())['config'] if done.exists() else None
    if made is None or any(made.get(k) != cfg.get(k) for k in DATA_KEYS):
        raise ValueError('shards were not completed under this data declaration')
    refuse_existing(args.output, *((args.model,) if args.command == 'train' else ()))
    started = time.monotonic()
    result = (train if args.command == 'train' else gate)(args, cfg)
    result.update({'command': args.command, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'seconds': time.monotonic() - started})
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=1))


if __name__ == '__main__':
    main()
