"""Train and gate a state-value MLP on re-encoded (encoder v2) value shards.

Input: the shard directory written by ``reencode_v2.py`` for the declaration in
``--config`` (e.g. ``experiments/value-v2-100k.json``). The model is an MLP on the
1,324 encoder-v2 features with BCE to "this seat won". The epoch is chosen on the
last ``validation_games`` train games. ``gate`` reads the test shards once and
reports pooled AUC for the network, the served leaf and the shared score, with a
game-clustered bootstrap of network AUC minus the better baseline. Provenance and
no-clobber rules from ``artifact_guard``; the data identity is the hash of the
shard manifests (which carry their source games' hashes).
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


def shards(root, split):
    files = sorted(Path(root).glob(f'{split}-*.npz'))
    if not files:
        raise ValueError(f'no {split} shards in {root}')
    data = [np.load(f) for f in files]
    out = {k: np.concatenate([d[k] for d in data]) for k in ('x', 'y', 'game', 'leaf', 'shared')}
    out['identity'] = hashlib.sha256(''.join(f.with_suffix('.json').read_text() for f in files).encode()).hexdigest()
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


def provenance(cfg, train_identity):
    return {'config': cfg, 'train_identity': train_identity, 'split': cfg['split']}


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
                                                **provenance(cfg, data['identity'])}, tmp))
    return {'fit_rows': len(x), 'val_rows': len(vx), 'best_epoch': best[1], 'history': history,
            'train_identity': data['identity'], 'model_sha256': sha256(args.model)}


def gate(args, cfg):
    import torch
    saved = torch.load(args.model, weights_only=False)
    train_identity = shards(args.data, 'train')['identity']
    if saved['config'] != cfg or saved['train_identity'] != train_identity:
        raise ValueError('model provenance does not match this declaration and training data')
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
            'test_identity': test['identity'], 'model_sha256': sha256(args.model), 'pass': lcb > 0}


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
