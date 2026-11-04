"""Train and gate the Catan policy/value (PV) network on the PV data (schema 2).

Shengji-style ``PolicyValueNet`` (AlphaGo/AlphaZero split of jobs): a shared trunk
over the pre-move observation, and per-candidate heads over the trunk plus the
candidate's action encoding, post-move observation and their difference:

- ``q`` (value of a candidate move): sigmoid, MSE on its playout win rate;
- ``p`` (policy logit): softmax over the decision's candidates, cross-entropy to a
  soft target ``softmax(win_rate / temperature)``;
- ``v`` (value of the position, "are we going to win from here"): sigmoid on the
  trunk, BCE on the game outcome, from the turn-start value rows.

Every constant comes from the config (``experiments/pv-train.json``), fixed before
the test split is read. ``train`` fits on the train split only, picks the epoch by
loss on its last ``validation_games`` games, and writes the model and a training
receipt. ``gate`` reads the test split once and reports the predeclared offline
bars. ``export`` writes plain-JSON weights for a pure-Python forward pass.

Torch and numpy are needed only here (host: ultra, ``~/catan-rank/.venv``).
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time


def load_games(root, lo, n, identity):
    games = []
    for seed in range(lo, lo + n):
        raw = (Path(root) / 'games' / f'{seed}.json').read_bytes()
        record = json.loads(raw)
        if record.get('seed') != seed or record.get('identity') != identity or record.get('schema') != 2:
            raise ValueError(f'record {seed} does not match the PV data identity')
        games.append((record, hashlib.sha256(raw).hexdigest()))
    return games


def tensors(games, torch):
    obs, cand, q, seg, served, rate, game_of = [], [], [], [], [], [], []
    vobs, vwon = [], []
    for g, (record, _) in enumerate(games):
        for row in record['value_rows']:
            vobs.append(row['features'])
            vwon.append(float(row['won']))
        for d in record['decisions']:
            if len(d['candidates']) < 2:
                continue
            i = len(obs)
            obs.append(d['obs'])
            game_of.append(g)
            for c in d['candidates']:
                delta = [a - b for a, b in zip(c['after'], d['obs'])]
                cand.append(c['act'] + c['after'] + delta)
                q.append(c['wins'] / c['playouts'])
                seg.append(i)
                served.append(float(c['served']))
                rate.append(c['wins'] / c['playouts'])
    f = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    return {'obs': f(obs), 'cand': f(cand), 'q': f(q), 'seg': torch.tensor(seg), 'served': f(served),
            'game': torch.tensor(game_of), 'vobs': f(vobs), 'vwon': f(vwon)}


def build(torch, obs_dim, cand_dim, hidden):
    nn = torch.nn

    class PVNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.q_head = nn.Sequential(nn.Linear(hidden + cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.p_head = nn.Sequential(nn.Linear(hidden + cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.v_head = nn.Linear(hidden, 1)

        def forward(self, obs, cand, seg):
            feat = self.trunk(obs)
            fc = torch.cat([feat[seg], cand], dim=-1)
            return self.q_head(fc).squeeze(-1), self.p_head(fc).squeeze(-1)

        def value(self, obs):
            return self.v_head(self.trunk(obs)).squeeze(-1)
    return PVNet()


def segment_softmax(torch, logits, seg, n):
    peak = torch.full((n,), -1e9).scatter_reduce(0, seg, logits, 'amax')
    ex = torch.exp(logits - peak[seg])
    return ex / torch.zeros(n).index_add(0, seg, ex)[seg]


def losses(torch, net, data, stats, cfg):
    obs = (data['obs'] - stats['obs_mean']) / stats['obs_sd']
    cand = (data['cand'] - stats['cand_mean']) / stats['cand_sd']
    q_logit, p_logit = net(obs, cand, data['seg'])
    n = len(data['obs'])
    q_loss = torch.mean((torch.sigmoid(q_logit) - data['q']) ** 2)
    target = segment_softmax(torch, data['q'] / cfg['policy_temperature'], data['seg'], n)
    prob = segment_softmax(torch, p_logit, data['seg'], n)
    p_loss = -torch.zeros(n).index_add(0, data['seg'], target * torch.log(prob + 1e-9)).mean()
    vobs = (data['vobs'] - stats['obs_mean']) / stats['obs_sd']
    v_loss = torch.nn.functional.binary_cross_entropy_with_logits(net.value(vobs), data['vwon'])
    w = cfg['loss_weights']
    return w['q'] * q_loss + w['p'] * p_loss + w['v'] * v_loss, (q_loss.item(), p_loss.item(), v_loss.item())


def train(args, cfg, data_cfg, manifest):
    import torch
    torch.manual_seed(cfg['seed'])
    lo, n = data_cfg['split']['train']
    games = load_games(args.data, lo, n, manifest['identity_sha256'])
    k = cfg['validation_games']
    fit, val = tensors(games[:-k], torch), tensors(games[-k:], torch)
    stats = {'obs_mean': fit['obs'].mean(0), 'obs_sd': fit['obs'].std(0).clamp_min(1e-6),
             'cand_mean': fit['cand'].mean(0), 'cand_sd': fit['cand'].std(0).clamp_min(1e-6)}
    net = build(torch, fit['obs'].shape[1], fit['cand'].shape[1], cfg['hidden'])
    opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    decisions = len(fit['obs'])
    best, history = None, []
    rng = random.Random(cfg['seed'])
    for epoch in range(cfg['epochs']):
        net.train()
        order = list(range(decisions))
        rng.shuffle(order)
        for start in range(0, decisions, cfg['batch_decisions']):
            pick = torch.tensor(sorted(order[start:start + cfg['batch_decisions']]))
            mask = torch.isin(fit['seg'], pick)
            remap = torch.full((decisions,), -1, dtype=torch.long)
            remap[pick] = torch.arange(len(pick))
            vpick = torch.randint(0, len(fit['vobs']), (cfg['batch_value_rows'],))
            batch = {'obs': fit['obs'][pick], 'cand': fit['cand'][mask], 'q': fit['q'][mask],
                     'seg': remap[fit['seg'][mask]], 'vobs': fit['vobs'][vpick], 'vwon': fit['vwon'][vpick]}
            loss, _ = losses(torch, net, batch, stats, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss, parts = losses(torch, net, val, stats, cfg)
        history.append({'epoch': epoch, 'val_loss': vloss.item(), 'q': parts[0], 'p': parts[1], 'v': parts[2]})
        if best is None or vloss.item() < best[0]:
            best = (vloss.item(), epoch, {k2: v.clone() for k2, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    torch.save({'state': net.state_dict(), 'stats': stats, 'config': cfg, 'epoch': best[1]}, args.model)
    return {'train_games': n, 'fit_decisions': decisions, 'val_decisions': len(val['obs']),
            'value_rows': len(fit['vobs']), 'best_epoch': best[1], 'history': history,
            'train_corpus_sha256': hashlib.sha256(''.join(h for _, h in games).encode()).hexdigest()}


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    rank, total, i = 0, 0., 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        tp = sum(y for _, y in pairs[i:j])
        total += tp * (rank + .5 * (j - i - tp))
        rank += (j - i) - tp
        i = j
    return total / (pos * neg) if pos and neg else None


def gate(args, cfg, data_cfg, manifest):
    import torch
    saved = torch.load(args.model, weights_only=False)
    stats = saved['stats']
    lo, n = data_cfg['split']['test']
    games = load_games(args.data, lo, n, manifest['identity_sha256'])
    test = tensors(games, torch)
    net = build(torch, test['obs'].shape[1], test['cand'].shape[1], cfg['hidden'])
    net.load_state_dict(saved['state'])
    net.eval()
    with torch.no_grad():
        q, p = net((test['obs'] - stats['obs_mean']) / stats['obs_sd'],
                   (test['cand'] - stats['cand_mean']) / stats['cand_sd'], test['seg'])
        v = net.value((test['vobs'] - stats['obs_mean']) / stats['obs_sd'])
    per = {}
    seg, rate, served = test['seg'].tolist(), test['q'].tolist(), test['served'].tolist()
    for idx, d in enumerate(seg):
        per.setdefault(d, []).append((q[idx].item(), p[idx].item(), rate[idx], served[idx]))
    rows = []
    for d, cands in per.items():
        served_rate = next(c[2] for c in cands if c[3])
        q_rate = max(cands, key=lambda c: c[0])[2]
        p_rate = max(cands, key=lambda c: c[1])[2]
        best_rate = max(c[2] for c in cands)
        rows.append((test['game'][d].item(), q_rate - served_rate, p_rate - served_rate, best_rate - served_rate))
    rng = random.Random(cfg['gate']['bootstrap_seed'])
    by_game = {}
    for r in rows:
        by_game.setdefault(r[0], []).append(r)
    ids = sorted(by_game)

    def mean(rs, i):
        return sum(r[i] for r in rs) / len(rs)
    draws = {1: [], 2: []}
    for _ in range(cfg['gate']['bootstrap_resamples']):
        sample = [r for gid in (rng.choice(ids) for _ in ids) for r in by_game[gid]]
        for i in draws:
            draws[i].append(mean(sample, i))
    for i in draws:
        draws[i].sort()
    lcb = {i: draws[i][int(.025 * len(draws[i]))] for i in draws}
    truncated = sum(c['truncated'] for record, _ in games for d in record['decisions'] for c in d['candidates'])
    v_auc = auc(v.tolist(), [int(x) for x in test['vwon'].tolist()])
    leaf_auc = auc([row['leaf'] for record, _ in games for row in record['value_rows']],
                   [int(row['won']) for record, _ in games for row in record['value_rows']])
    passed = truncated == 0 and lcb[1] > 0
    return {'test_games': n, 'test_decisions': len(rows), 'truncated_test_playouts': truncated,
            'q_pick_minus_served': mean(rows, 1), 'q_pick_lcb95': lcb[1],
            'p_pick_minus_served': mean(rows, 2), 'p_pick_lcb95': lcb[2],
            'oracle_best_minus_served': mean(rows, 3),
            'served_pick_is_playout_best_share': sum(r[3] == 0 for r in rows) / len(rows),
            'v_auc': v_auc, 'served_leaf_auc_on_value_rows': leaf_auc,
            'test_corpus_sha256': hashlib.sha256(''.join(h for _, h in games).encode()).hexdigest(),
            'pass': passed}


def export(args):
    import torch
    saved = torch.load(args.model, weights_only=False)
    out = {'schema': 'catan-pv-v1', 'config': saved['config'], 'epoch': saved['epoch'],
           'stats': {k: v.tolist() for k, v in saved['stats'].items()},
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
    if args.command == 'export':
        return export(args)
    started = time.monotonic()
    result = (train if args.command == 'train' else gate)(args, cfg, data_cfg, manifest)
    result.update({'command': args.command, 'config': cfg, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   'data_identity_sha256': manifest['identity_sha256'],
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'seconds': time.monotonic() - started})
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'config')}, indent=1))


if __name__ == '__main__':
    main()
