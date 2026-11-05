"""Train and gate the PV v2 afterstate value model (model audit fixes 2, 3 and 5).

Rows are the candidates of ``scripts/pv2_data.py`` decisions: encoder-v2 features
of the afterstate (``x``) and the candidate's playout win rate. The model is an MLP
whose sigmoid output is "my win chance after this turn". Loss, per batch of whole
decisions: soft binary cross-entropy to the playout win rate, plus ``relative_weight``
x MSE between each candidate's prediction minus its decision's mean and the same
for the playout rates (the within-decision ordering the search consumes).

``gate`` (test split, more playouts per candidate): at each decision the model
picks its best candidate (an immediate win is picked first); the bar is the
game-clustered bootstrap 95% lower bound of (picked candidate's playout rate minus
the served candidate's) > 0. Reported, not gated: oracle-best minus served, how
often the served pick is playout-best, and on decisions with both "end now" and a
city-building candidate, how often the model and the playouts rank the city higher.

Provenance and no-clobber rules from ``artifact_guard``. Torch only here.
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


def load(root, lo, n, identity):
    games, digests = [], []
    for seed in range(lo, lo + n):
        raw = (Path(root) / 'games' / f'{seed}.json').read_bytes()
        record = json.loads(raw)
        if record.get('seed') != seed or record.get('identity') != identity or record.get('schema') != 'pv2':
            raise ValueError(f'record {seed} does not match the PV2 data identity')
        games.append(record)
        digests.append(hashlib.sha256(raw).hexdigest())
    return games, hashlib.sha256(''.join(digests).encode()).hexdigest()


def decisions(games):
    out = []
    for g, record in enumerate(games):
        for d in record['decisions']:
            cands = [c for c in d['candidates'] if c['playouts'] > 0 or c['won_now']]
            if len(cands) >= 2 and any(c['served'] for c in cands):
                out.append((g, cands))
    return out


def to_tensors(torch, items):
    xs, rates, seg = [], [], []
    for i, (_, cands) in enumerate(items):
        for c in cands:
            if c['won_now']:
                continue
            xs.append(c['x'])
            rates.append(c['wins'] / c['playouts'])
            seg.append(i)
    return torch.tensor(xs), torch.tensor(rates), torch.tensor(seg), len(items)


def build(torch, dim, hidden, layers):
    mods, width = [], dim
    for _ in range(layers):
        mods += [torch.nn.Linear(width, hidden), torch.nn.ReLU()]
        width = hidden
    mods.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*mods)


def batch_loss(torch, net, x, rate, seg, n, cfg):
    z = net(x).squeeze(-1)
    p = torch.sigmoid(z)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, rate)
    count = torch.zeros(n).index_add(0, seg, torch.ones_like(p)).clamp_min(1)
    pm = torch.zeros(n).index_add(0, seg, p) / count
    rm = torch.zeros(n).index_add(0, seg, rate) / count
    rel = torch.mean(((p - pm[seg]) - (rate - rm[seg])) ** 2)
    return bce + cfg['relative_weight'] * rel, (bce.item(), rel.item())


def train(args, cfg, data_cfg, manifest):
    import torch
    torch.manual_seed(cfg['seed'])
    lo, n = data_cfg['split']['train']
    games, corpus = load(args.data, lo, n, manifest['identity_sha256'])
    k = cfg['validation_games']
    fit_items, val_items = decisions(games[:-k]), decisions(games[-k:])
    x, rate, seg, nd = to_tensors(torch, fit_items)
    vx, vrate, vseg, vnd = to_tensors(torch, val_items)
    mean, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    x, vx = (x - mean) / sd, (vx - mean) / sd
    net = build(torch, x.shape[1], cfg['hidden'], cfg['layers'])
    opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = random.Random(cfg['seed'])
    best, history = None, []
    for epoch in range(cfg['epochs']):
        net.train()
        order = list(range(nd))
        rng.shuffle(order)
        for start in range(0, nd, cfg['batch_decisions']):
            pick = torch.tensor(sorted(order[start:start + cfg['batch_decisions']]))
            mask = torch.isin(seg, pick)
            remap = torch.full((nd,), -1, dtype=torch.long)
            remap[pick] = torch.arange(len(pick))
            loss, _ = batch_loss(torch, net, x[mask], rate[mask], remap[seg[mask]], len(pick), cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss, parts = batch_loss(torch, net, vx, vrate, vseg, vnd, cfg)
        history.append({'epoch': epoch, 'val_loss': vloss.item(), 'bce': parts[0], 'relative': parts[1]})
        if best is None or vloss.item() < best[0]:
            best = (vloss.item(), epoch, {key: v.clone() for key, v in net.state_dict().items()})
    publish(args.model, lambda tmp: torch.save({'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1],
                                                **provenance(cfg, data_cfg, manifest)}, tmp))
    return {'train_games': n, 'fit_decisions': nd, 'val_decisions': vnd, 'fit_rows': len(x), 'best_epoch': best[1],
            'history': history, 'train_corpus_sha256': corpus, 'model_sha256': sha256(args.model)}


def gate(args, cfg, data_cfg, manifest):
    import torch
    saved = torch.load(args.model, weights_only=False)
    validate(saved, cfg, data_cfg, manifest)
    lo, n = data_cfg['split']['test']
    games, corpus = load(args.data, lo, n, manifest['identity_sha256'])
    items = decisions(games)
    net = build(torch, len(saved['mean']), cfg['hidden'], cfg['layers'])
    net.load_state_dict(saved['state'])
    net.eval()
    rows, city_model, city_play, city_n = [], 0, 0, 0
    with torch.no_grad():
        for g, cands in items:
            scores = []
            for c in cands:
                if c['won_now']:
                    scores.append(float('inf'))
                else:
                    x = (torch.tensor([c['x']]) - saved['mean']) / saved['sd']
                    scores.append(net(x).item())
            rate = [c['wins'] / c['playouts'] if c['playouts'] else 1. for c in cands]
            served = next(i for i, c in enumerate(cands) if c['served'])
            picked = max(range(len(cands)), key=lambda i: (scores[i], -i))
            rows.append((g, rate[picked] - rate[served], max(rate) - rate[served], float(rate[served] == max(rate))))
            end = next((i for i, c in enumerate(cands) if not c['seq']), None)
            city = next((i for i, c in enumerate(cands) if any(a['type'] == 'build_city' for a in c['seq'])), None)
            if end is not None and city is not None:
                city_n += 1
                city_model += scores[city] > scores[end]
                city_play += rate[city] > rate[end]
    by_game = {}
    for r in rows:
        by_game.setdefault(r[0], []).append(r)
    ids = sorted(by_game)
    rng = random.Random(cfg['gate']['bootstrap_seed'])
    draws = []
    for _ in range(cfg['gate']['bootstrap_resamples']):
        sample = [r for gid in (rng.choice(ids) for _ in ids) for r in by_game[gid]]
        draws.append(sum(r[1] for r in sample) / len(sample))
    draws.sort()
    lcb = draws[int(.025 * len(draws))]
    gain = sum(r[1] for r in rows) / len(rows)
    return {'test_games': n, 'test_decisions': len(rows), 'pick_minus_served': gain, 'pick_minus_served_lcb95': lcb,
            'oracle_minus_served': sum(r[2] for r in rows) / len(rows),
            'served_is_playout_best_share': sum(r[3] for r in rows) / len(rows),
            'city_vs_end_decisions': city_n,
            'city_ranked_above_end_by_model': city_model / city_n if city_n else None,
            'city_above_end_in_playouts': city_play / city_n if city_n else None,
            'test_corpus_sha256': corpus, 'model_sha256': sha256(args.model),
            'pass': lcb > cfg['gate']['minimum_gain_lcb']}


def export(args, cfg, data_cfg, manifest):
    import torch
    saved = torch.load(args.model, weights_only=False)
    validate(saved, cfg, data_cfg, manifest)
    out = {'schema': 'catan-pv2-value-v1', 'epoch': saved['epoch'], 'config': saved['config'],
           'model_sha256': sha256(args.model), 'data_identity_sha256': saved['data_identity_sha256'],
           'train_split': saved['train_split'], 'mean': saved['mean'].tolist(), 'sd': saved['sd'].tolist(),
           'state': {k: v.tolist() for k, v in saved['state'].items()}}
    with Path(args.output).open('x') as stream:
        json.dump(out, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'gate', 'export'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    data_cfg = json.loads(args.data_config.read_text())
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if manifest['config'] != data_cfg:
        raise ValueError('data root was produced with a different data config')
    refuse_existing(args.output, *((args.model,) if args.command == 'train' else ()))
    if args.command == 'export':
        return export(args, cfg, data_cfg, manifest)
    import torch
    torch.set_num_threads(cfg.get('threads', 8))
    started = time.monotonic()
    result = (train if args.command == 'train' else gate)(args, cfg, data_cfg, manifest)
    result.update({'command': args.command, 'config': cfg,
                   'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   'data_identity_sha256': manifest['identity_sha256'],
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'seconds': time.monotonic() - started})
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'config')}, indent=1))


if __name__ == '__main__':
    main()
