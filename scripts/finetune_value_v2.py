"""Fine-tune the encoder-v2 state-value net on all current data (no new data generation).

Starts from a trained net (``--base-model``, e.g. arm B ``value-v2-100k-aug-b``) and
keeps its symmetric standardization. Each step mixes:

- ``shards``: a batch of 100k-game end-of-turn rows (BCE to the outcome);
- ``visited``: a batch of pv2b visited end-of-turn rows, including explored turns
  (BCE to the outcome) — the off-policy states the 100k data lacks;
- ``decisions``: a batch of pv2b counterfactual decisions: within-decision relative
  loss (prediction minus decision mean vs playout rate minus decision mean) plus a
  soft BCE to the playout rate.

Board-symmetry augmentation on every row. pv2b validation games (the last
``validation_games`` of its train split) are excluded from training; after each
epoch the within-decision relative loss on them is recorded and the epoch with the
lowest value is kept (predeclared selection rule). pv2b test games are not read.
Exports ``<out>.npz`` + ``<out>.meta.json`` with a torch parity probe.
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
from artifact_guard import refuse_existing, sha256  # noqa: E402
from train_pv2 import decisions as pv2_decisions, load as pv2_load  # noqa: E402
from train_value_v2 import build_model, make_optimizer, shards, symmetry_index  # noqa: E402


def visited_rows(games):
    xs, ys = [], []
    for g in games:
        for r in g['value_rows']:
            if r['point'] == 'end_turn':
                xs.append(r['v2'])
                ys.append(float(r['won']))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def decision_tensors(torch, items):
    xs, rate, seg = [], [], []
    for i, (_, cands) in enumerate(items):
        for c in cands:
            if c['won_now']:
                continue
            xs.append(c['x'])
            rate.append(c['wins'] / c['playouts'])
            seg.append(i)
    return torch.tensor(xs), torch.tensor(rate), torch.tensor(seg), len(items)


def relative_loss(torch, z, rate, seg, n):
    p = torch.sigmoid(z)
    count = torch.zeros(n).index_add(0, seg, torch.ones_like(p)).clamp_min(1)
    pm = torch.zeros(n).index_add(0, seg, p) / count
    rm = torch.zeros(n).index_add(0, seg, rate) / count
    return torch.mean(((p - pm[seg]) - (rate - rm[seg])) ** 2)


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--pv2-data', type=Path, required=True)
    parser.add_argument('--pv2-data-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='output stem (writes .pt, .npz, .meta.json, .json)')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    base_cfg = json.loads(args.base_config.read_text())
    dcfg = json.loads(args.pv2_data_config.read_text())
    outs = [args.out.with_suffix(s) for s in ('.pt', '.npz', '.meta.json', '.json')]
    refuse_existing(*outs)
    manifest = json.loads((args.pv2_data / 'manifest.json').read_text())
    if manifest['config'] != dcfg:
        raise ValueError('pv2 data root was produced with a different data config')
    m = cfg['finetune']
    torch.manual_seed(m['seed'])
    torch.set_num_threads(m['threads'])
    rng = random.Random(m['seed'])
    started = time.monotonic()
    base = torch.load(args.base_model, weights_only=False)
    if base['config'] != base_cfg:
        raise ValueError('base model was not trained under the base config')
    net = build_model(torch, len(base['mean']), base_cfg['model'])
    net.load_state_dict(base['state'])
    mean, sd = base['mean'], base['sd']
    perms = symmetry_index(torch)
    # data
    S = shards(args.shards, 'train')
    lo, n = base_cfg['split']['train']
    s_fit = S['game'] < lo + n - base_cfg['model']['validation_games']
    sx, sy = S['x'][s_fit], S['y'][s_fit].astype(np.float32)
    tlo, tn = dcfg['split']['train']
    k = cfg['validation_games']
    games, corpus = pv2_load(args.pv2_data, tlo, tn, manifest['identity_sha256'])
    fit_games, val_games = games[:-k], games[-k:]
    vx_np, vy_np = visited_rows(fit_games)
    dfit = pv2_decisions(fit_games)
    dval = pv2_decisions(val_games)
    dx, drate, dseg, dn = decision_tensors(torch, dfit)
    vdx, vdrate, vdseg, vdn = decision_tensors(torch, dval)

    def std(x):
        return (torch.as_tensor(x, dtype=torch.float32) - mean) / sd

    def aug(xb):
        return torch.gather(xb, 1, perms[torch.randint(0, len(perms), (len(xb),))])
    dx, vdx = std(dx), std(vdx)
    steps_per_epoch = (dn + m['batch_decisions'] - 1) // m['batch_decisions']
    opt, sched = make_optimizer(torch, net, m, m['epochs'] * steps_per_epoch)
    bce = torch.nn.BCEWithLogitsLoss()
    best, history = None, []
    for epoch in range(m['epochs']):
        net.train()
        order = list(range(dn))
        rng.shuffle(order)
        for start in range(0, dn, m['batch_decisions']):
            pick = torch.tensor(sorted(order[start:start + m['batch_decisions']]))
            mask = torch.isin(dseg, pick)
            remap = torch.full((dn,), -1, dtype=torch.long)
            remap[pick] = torch.arange(len(pick))
            zd = net(aug(dx[mask])).squeeze(-1)
            loss = m['relative_weight'] * relative_loss(torch, zd, drate[mask], remap[dseg[mask]], len(pick))
            loss = loss + m['decision_bce_weight'] * torch.nn.functional.binary_cross_entropy_with_logits(zd, drate[mask])
            si = np.random.default_rng(rng.randrange(1 << 30)).integers(0, len(sx), m['batch_shards'])
            loss = loss + m['shard_weight'] * bce(net(aug(std(sx[si]))).squeeze(-1), torch.from_numpy(sy[si]))
            vi = np.random.default_rng(rng.randrange(1 << 30)).integers(0, len(vx_np), m['batch_visited'])
            loss = loss + m['visited_weight'] * bce(net(aug(std(vx_np[vi]))).squeeze(-1), torch.from_numpy(vy_np[vi]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        net.eval()
        with torch.no_grad():
            zv = net(vdx).squeeze(-1)
            rel = relative_loss(torch, zv, vdrate, vdseg, vdn).item()
            # descriptive: picked minus served on validation decisions
            gains, i = [], 0
            for _, cands in dval:
                rows = [c for c in cands if not c['won_now']]
                scores = zv[i:i + len(rows)].tolist()
                i += len(rows)
                it = iter(scores)
                full = [float('inf') if c['won_now'] else next(it) for c in cands]
                rate = [c['wins'] / c['playouts'] if c['playouts'] else 1. for c in cands]
                served = next(j for j, c in enumerate(cands) if c['served'])
                picked = max(range(len(cands)), key=lambda j: (full[j], -j))
                gains.append(rate[picked] - rate[served])
        history.append({'epoch': epoch, 'val_relative': rel, 'val_pick_minus_served': sum(gains) / len(gains)})
        print(json.dumps(history[-1]), flush=True)
        if best is None or rel < best[0]:
            best = (rel, epoch, {key: v.clone() for key, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    torch.save({'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1], 'config': cfg, 'base_config': base_cfg,
                'base_model_sha256': sha256(args.base_model), 'pv2_identity': manifest['identity_sha256']}, outs[0])
    arrays = {('w:' + key): v.numpy().astype(np.float32) for key, v in best[2].items()}
    arrays.update(mean=mean.numpy().astype(np.float32), sd=sd.numpy().astype(np.float32))
    probe = torch.from_numpy(sx[:64].astype(np.float32))
    with torch.no_grad():
        arrays['parity_x'] = probe.numpy()
        arrays['parity_logit'] = net((probe - mean) / sd).squeeze(-1).numpy()
    with outs[1].open('xb') as stream:
        np.savez(stream, **arrays)
    meta = {'schema': 'catan-value-v2-resmlp-v1', 'arch': base_cfg['model'], 'model_sha256': sha256(outs[0]),
            'config_name': cfg['name'], 'epoch': best[1], 'npz_sha256': sha256(outs[1])}
    outs[2].write_text(json.dumps(meta, indent=1))
    receipt = {'config': cfg, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
               'base_model_sha256': sha256(args.base_model), 'shard_rows': int(len(sx)), 'visited_rows': int(len(vx_np)),
               'fit_decisions': dn, 'val_decisions': vdn, 'best_epoch': best[1], 'history': history,
               'pv2_train_corpus_sha256': corpus, 'npz_sha256': meta['npz_sha256'],
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'seconds': time.monotonic() - started}
    with outs[3].open('x') as stream:
        json.dump(receipt, stream, indent=1)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('history', 'config')}, indent=1))


if __name__ == '__main__':
    main()
