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
    mean, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    x, vx = (x - mean) / sd, (vx - mean) / sd
    layers, width = [], x.shape[1]
    for _ in range(m['layers']):
        layers += [torch.nn.Linear(width, m['hidden']), torch.nn.ReLU()]
        width = m['hidden']
    net = torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=m['learning_rate'], weight_decay=m['weight_decay'])
    lossf = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(m['seed'])
    best, history = None, []
    for epoch in range(m['epochs']):
        net.train()
        for idx in torch.randperm(len(x), generator=gen).split(m['batch']):
            loss = lossf(net(x[idx]).squeeze(-1), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
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
    m = cfg['model']
    layers, width = [], test['x'].shape[1]
    for _ in range(m['layers']):
        layers += [torch.nn.Linear(width, m['hidden']), torch.nn.ReLU()]
        width = m['hidden']
    net = torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))
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
    if not (Path(args.data) / 'DONE.json').exists() or json.loads((Path(args.data) / 'DONE.json').read_text())['config'] != cfg:
        raise ValueError('shards were not completed under this declaration')
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
