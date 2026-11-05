"""Train and gate value heads ("are we going to win from here") on value-row data.

Board row 4 follow-up (Jerry 2026-11-04: retrain on the 10k-game value data).
For each sampling point in the config (``roll`` = turn start, ``end_turn`` = after
the seat's own moves), fit an MLP on the 40 observation-only features with
binary cross-entropy to the game outcome. Every constant comes from the config,
fixed before the test split is read. The epoch is chosen on the last
``validation_games`` train games. ``gate`` reads the test split once and reports,
per point, AUC for the network, the served leaf and the served shared score, and a
game-clustered bootstrap of network AUC minus the better baseline. ``export``
writes plain-JSON weights for a pure-Python forward pass. Torch only here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_guard import provenance, publish, refuse_existing, sha256, validate  # noqa: E402


def load(root, lo, n, identity, point):
    feats, won, leaf, shared, game, digests = [], [], [], [], [], []
    for g, seed in enumerate(range(lo, lo + n)):
        raw = (Path(root) / 'games' / f'{seed}.json').read_bytes()
        record = json.loads(raw)
        if record.get('seed') != seed or record.get('identity') != identity:
            raise ValueError(f'record {seed} does not match the data identity')
        digests.append(hashlib.sha256(raw).hexdigest())
        for row in record['value_rows']:
            if row.get('point', 'roll') != point:
                continue
            feats.append(row['features'])
            won.append(float(row['won']))
            leaf.append(row['leaf'])
            shared.append(row['shared'])
            game.append(g)
    return {'x': feats, 'y': won, 'leaf': leaf, 'shared': shared, 'game': game,
            'corpus_sha256': hashlib.sha256(''.join(digests).encode()).hexdigest()}


def build(torch, dim, hidden, layers):
    mods, width = [], dim
    for _ in range(layers):
        mods += [torch.nn.Linear(width, hidden), torch.nn.ReLU()]
        width = hidden
    mods.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*mods)


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    below, total, i = 0, 0., 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        tp = sum(y for _, y in pairs[i:j])
        total += tp * (below + .5 * ((j - i) - tp))
        below += (j - i) - tp
        i = j
    return total / (pos * neg) if pos and neg else None


def train_point(torch, args, cfg, data_cfg, manifest, point):
    torch.manual_seed(cfg['seed'])
    lo, n = data_cfg['split']['train']
    k = cfg['validation_games']
    fit = load(args.data, lo, n - k, manifest['identity_sha256'], point)
    val = load(args.data, lo + n - k, k, manifest['identity_sha256'], point)
    x, y = torch.tensor(fit['x']), torch.tensor(fit['y'])
    mean, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    vx, vy = (torch.tensor(val['x']) - mean) / sd, torch.tensor(val['y'])
    x = (x - mean) / sd
    net = build(torch, x.shape[1], cfg['hidden'], cfg['layers'])
    opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    lossf = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(cfg['seed'])
    best, history = None, []
    for epoch in range(cfg['epochs']):
        net.train()
        order = torch.randperm(len(x), generator=gen)
        for start in range(0, len(x), cfg['batch']):
            idx = order[start:start + cfg['batch']]
            loss = lossf(net(x[idx]).squeeze(-1), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(vx).squeeze(-1), vy).item()
        history.append({'epoch': epoch, 'val_loss': vloss})
        if best is None or vloss < best[0]:
            best = (vloss, epoch, {key: v.clone() for key, v in net.state_dict().items()})
    path = Path(args.model_dir) / f'value-{point}.pt'
    publish(path, lambda tmp: torch.save({'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1],
                                          **provenance(cfg, data_cfg, manifest, point)}, tmp))
    return {'fit_rows': len(x), 'val_rows': len(vx), 'best_epoch': best[1], 'history': history,
            'model_sha256': sha256(path),
            'fit_corpus_sha256': fit['corpus_sha256'], 'val_corpus_sha256': val['corpus_sha256']}


def gate_point(torch, args, cfg, data_cfg, manifest, point):
    path = Path(args.model_dir) / f'value-{point}.pt'
    saved = torch.load(path, weights_only=False)
    validate(saved, cfg, data_cfg, manifest, point)
    lo, n = data_cfg['split']['test']
    test = load(args.data, lo, n, manifest['identity_sha256'], point)
    net = build(torch, len(test['x'][0]), cfg['hidden'], cfg['layers'])
    net.load_state_dict(saved['state'])
    net.eval()
    with torch.no_grad():
        pred = net((torch.tensor(test['x']) - saved['mean']) / saved['sd']).squeeze(-1).tolist()
    y = [int(v) for v in test['y']]
    by_game = {}
    for i, g in enumerate(test['game']):
        by_game.setdefault(g, []).append(i)
    ids = sorted(by_game)

    def metrics(index):
        lab = [y[i] for i in index]
        return {'net': auc([pred[i] for i in index], lab), 'leaf': auc([test['leaf'][i] for i in index], lab),
                'shared': auc([test['shared'][i] for i in index], lab)}
    point_m = metrics(range(len(y)))
    rng = random.Random(cfg['gate']['bootstrap_seed'])
    diffs = []
    for _ in range(cfg['gate']['bootstrap_resamples']):
        sample = [i for gid in (rng.choice(ids) for _ in ids) for i in by_game[gid]]
        m = metrics(sample)
        diffs.append(m['net'] - max(m['leaf'], m['shared']))
    diffs.sort()
    lcb = diffs[int(.025 * len(diffs))]
    return {'test_rows': len(y), 'auc': point_m, 'gain_vs_best_baseline': point_m['net'] - max(point_m['leaf'], point_m['shared']),
            'gain_lcb95': lcb, 'pass': lcb > cfg['gate']['minimum_gain_lcb'],
            'test_corpus_sha256': test['corpus_sha256'], 'epoch': saved['epoch'], 'model_sha256': sha256(path)}


def export(args, point, cfg, data_cfg, manifest):
    import torch
    path = Path(args.model_dir) / f'value-{point}.pt'
    saved = torch.load(path, weights_only=False)
    validate(saved, cfg, data_cfg, manifest, point)
    out = {'schema': 'catan-value-mlp-v1', 'point': point, 'epoch': saved['epoch'], 'config': saved['config'],
           'model_sha256': sha256(path), 'data_identity_sha256': saved['data_identity_sha256'],
           'train_split': saved['train_split'],
           'mean': saved['mean'].tolist(), 'sd': saved['sd'].tolist(),
           'state': {k: v.tolist() for k, v in saved['state'].items()}}
    with (Path(args.model_dir) / f'value-{point}.json').open('x') as stream:
        json.dump(out, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'gate', 'export'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads(args.data_config.read_text())
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if manifest['config'] != data_cfg:
        raise ValueError('data root was produced with a different data config')
    if args.command == 'export':
        refuse_existing(*(Path(args.model_dir) / f'value-{p}.json' for p in cfg['points']))
        for point in cfg['points']:
            export(args, point, cfg, data_cfg, manifest)
        return
    if args.command == 'train':
        refuse_existing(args.output, *(Path(args.model_dir) / f'value-{p}.pt' for p in cfg['points']))
    else:
        refuse_existing(args.output)
    import torch
    torch.set_num_threads(cfg.get('threads', 8))
    args.model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    step = train_point if args.command == 'train' else gate_point
    result = {point: step(torch, args, cfg, data_cfg, manifest, point) for point in cfg['points']}
    result = {'command': args.command, 'points': result, 'config': cfg,
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'data_identity_sha256': manifest['identity_sha256'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds': time.monotonic() - started}
    if args.command == 'gate':
        result['pass'] = all(r['pass'] for r in result['points'].values())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({p: {k: v for k, v in r.items() if k != 'history'} for p, r in result['points'].items()}, indent=1))


if __name__ == '__main__':
    main()
