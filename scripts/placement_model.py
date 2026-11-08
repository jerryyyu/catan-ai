"""Placement model on placement-data-v1 (experiments/placement-model.json).

A tile-token transformer (``arch_bakeoff.make_net('tiles')``) whose global-token
projection also takes 21 extra inputs per candidate row: the opponents' public
style posterior (3 x 4, seat-relative), a setup_index one-hot (8) and
served_rank / 16. Trained on whole decisions (16 candidates each) with soft BCE
to wins/playouts plus 1.0 x a within-decision listwise cross-entropy; the epoch
is chosen by the validation listwise loss on the last 45 train games. The gate
reads the 50 test games once: per decision, the model's top candidate's playout
rate minus the served (served_rank 0) candidate's rate, with a game-clustered
bootstrap 95% lower bound > 0.

    placement_model.py train --config experiments/placement-model.json \
        --data runs/placement-data-v1 --out runs/placement-model
    placement_model.py gate --config experiments/placement-model.json \
        --data runs/placement-data-v1 --out runs/placement-model

``--max-games`` / ``--epochs`` are smoke overrides; their outputs carry
``smoke: true`` and the gate refuses such a checkpoint unless ``--allow-smoke``.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_bakeoff as ab  # noqa: E402
from arch_bakeoff import fast_auc, layout, make_net, standardization  # noqa: E402
from artifact_guard import publish, refuse_existing, sha256  # noqa: E402
from train_value_v2 import make_optimizer, symmetry_index  # noqa: E402

N_CANDIDATES, N_OPPONENTS, N_STYLES, N_SETUP = 16, 3, 4, 8
EXTRA_W = N_OPPONENTS * N_STYLES + N_SETUP + 1  # 21
LISTWISE_W = 1.0
EVAL_BATCH = 4096


# --------------------------------------------------------------------- data
def split_ranges(cfg):
    return {k: (int(lo), int(n)) for k, (lo, n) in cfg['data']['split'].items()}


def validation_games(cfg):
    m = re.search(r'last (\d+) train games \((\d+)-(\d+)\)', cfg['data']['validation'])
    if not m:
        raise ValueError('cannot read the validation games from data.validation')
    k, a, b = map(int, m.groups())
    lo, n = split_ranges(cfg)['train']
    if (a, b) != (lo + n - k, lo + n - 1):
        raise ValueError('data.validation range disagrees with the train split')
    return k


def selected_seeds(cfg, split, max_games=None):
    """The declared seeds of ``split``. For a smoke train (``max_games``), the first fit
    games and the first validation games, in the declared 405:45 proportion."""
    lo, n = split_ranges(cfg)[split]
    seeds = list(range(lo, lo + n))
    if split != 'train' or not max_games or max_games >= n:
        return seeds
    k = validation_games(cfg)
    nv = max(1, round(max_games * k / n))
    return seeds[:max_games - nv] + seeds[n - k:n - k + nv]


def check_names(root, cfg):
    """Every game file must belong to the declared split."""
    allowed = set()
    for lo, n in split_ranges(cfg).values():
        allowed |= set(range(lo, lo + n))
    for path in sorted((Path(root) / 'games').glob('*.json')):
        if not path.stem.isdigit() or int(path.stem) not in allowed:
            raise ValueError(f'game file {path.name} is outside the declared split')


def identity(hashes):
    return hashlib.sha256(json.dumps({str(k): v for k, v in sorted(hashes.items())}).encode()).hexdigest()


def file_hashes(root, seeds):
    return {s: sha256(Path(root) / 'games' / f'{s}.json') for s in seeds}


def load(root, cfg, split, max_games=None):
    """Rows of the selected games of ``split``, validated, contiguous by decision."""
    check_names(root, cfg)
    seeds = selected_seeds(cfg, split, max_games)
    lo, n = split_ranges(cfg)[split]
    xs, extra, wins, playouts, rank, game, dec_start, dec_meta, hashes = [], [], [], [], [], [], [], [], {}
    counts = {}
    row = 0
    for s in seeds:
        path = Path(root) / 'games' / f'{s}.json'
        if not path.exists():
            raise ValueError(f'missing game file {path.name}')
        raw = path.read_bytes()
        hashes[s] = hashlib.sha256(raw).hexdigest()
        g = json.loads(raw)
        if g.get('seed') != s:
            raise ValueError(f'{path.name} holds seed {g.get("seed")}')
        if not lo <= s < lo + n:
            raise ValueError(f'seed {s} is outside the declared {split} split')
        for d in g['decisions']:
            cands = d['candidates']
            counts[len(cands)] = counts.get(len(cands), 0) + 1
            if not 1 <= len(cands) <= N_CANDIDATES:
                raise ValueError(f'seed {s}: decision with {len(cands)} candidates')
            if sum(c['served_rank'] == 0 for c in cands) != 1:
                raise ValueError(f'seed {s}: served_rank 0 is not exactly one candidate')
            if any(c['served'] != (c['served_rank'] == 0) for c in cands):
                raise ValueError(f'seed {s}: served flag disagrees with served_rank 0')
            post = np.asarray(d['posterior'], dtype=np.float64)
            if post.shape != (N_OPPONENTS, N_STYLES) or np.any(np.abs(post.sum(1) - 1) > 1e-6) or np.any(post < 0):
                raise ValueError(f'seed {s}: posterior rows must be 3 x 4 and sum to 1')
            if not 0 <= d['setup_index'] < N_SETUP:
                raise ValueError(f'seed {s}: setup_index {d["setup_index"]} out of range')
            onehot = np.zeros(N_SETUP)
            onehot[d['setup_index']] = 1
            dec_start.append(row)
            dec_meta.append((s, d['seat'], d['setup_index']))
            for c in cands:
                if len(c['x']) != ab.WIDTH:
                    raise ValueError(f'seed {s}: x has length {len(c["x"])}, not {ab.WIDTH}')
                if not 0 <= c['wins'] <= c['playouts'] or c['playouts'] <= 0:
                    raise ValueError(f'seed {s}: bad wins/playouts')
                xs.append(np.asarray(c['x'], dtype=np.float32))
                extra.append(np.concatenate([post.ravel(), onehot, [c['served_rank'] / N_CANDIDATES]]))
                wins.append(c['wins'])
                playouts.append(c['playouts'])
                rank.append(c['served_rank'])
                game.append(s)
                row += 1
    if not dec_start:
        raise ValueError(f'no decisions in the {split} split')
    meta = np.asarray(dec_meta, dtype=np.int64)
    return {'x': np.stack(xs), 'extra': np.asarray(extra, dtype=np.float32),
            'wins': np.asarray(wins, dtype=np.int64), 'playouts': np.asarray(playouts, dtype=np.int64),
            'rate': np.asarray(wins, dtype=np.float64) / np.asarray(playouts, dtype=np.float64),
            'rank': np.asarray(rank, dtype=np.int64), 'game': np.asarray(game, dtype=np.int64),
            'start': np.asarray(dec_start + [row], dtype=np.int64),
            'dec_game': meta[:, 0], 'dec_seat': meta[:, 1], 'dec_setup': meta[:, 2],
            'seeds': seeds, 'hashes': hashes, 'identity': identity(hashes),
            'candidate_counts': {str(k): v for k, v in sorted(counts.items())}}


# -------------------------------------------------------------------- model
def check_model_config(cfg):
    m = cfg['model']
    want = {'d': ab.TILES_D, 'layers': ab.TILES_LAYERS, 'heads': ab.TILES_HEADS, 'mlp': ab.TILES_MLP,
            'dropout': ab.TILES_DROPOUT}
    if any(m[k] != v for k, v in want.items()):
        raise ValueError('model sizes differ from arch_bakeoff tiles constants')


def make_model(torch, board=None):
    """``arch_bakeoff`` tiles net with the global-token projection widened to take the
    21 extra inputs (concatenated after the 20 encoder-v2 global features)."""
    nn = torch.nn

    class PlacementNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = make_net(torch, 'tiles', board)
            self.core.glob = nn.Linear(ab.GLOBAL_W + EXTRA_W, ab.TILES_D)

        def forward(self, x, extra):
            c, n = self.core, len(x)
            h = torch.cat([c.tile(x[:, c.tidx]),
                           c.player(x[:, ab.P0:ab.G0].reshape(n, ab.N_PLAYERS, ab.PLAYER_W)),
                           c.glob(torch.cat([x[:, ab.G0:], extra], 1)).unsqueeze(1)], 1) + c.pos + c.kind[c.types]
            return c.head(c.norm(c.encoder(h)[:, -1])).squeeze(-1)

    return PlacementNet()


# --------------------------------------------------------------------- loss
def decision_index(torch, starts, device):
    """For decisions with row ranges ``starts[i]:starts[i+1]`` *relative to the batch*:
    a [D, K] row index (pad = 0) and a [D, K] mask."""
    sizes = np.diff(starts)
    k = int(sizes.max())
    idx = np.zeros((len(sizes), k), dtype=np.int64)
    mask = np.zeros((len(sizes), k), dtype=bool)
    for i, (a, m) in enumerate(zip(starts[:-1], sizes)):
        idx[i, :m] = np.arange(a, a + m)
        mask[i, :m] = True
    return torch.from_numpy(idx).to(device), torch.from_numpy(mask).to(device)


def listwise_terms(torch, logits, rate, idx, mask):
    """Per decision: -sum q log softmax(logits) with q = rates / sum(rates), and whether the
    decision has any nonzero rate (all-zero decisions get no listwise term)."""
    lg = logits[idx].masked_fill(~mask, float('-inf'))
    r = rate[idx].masked_fill(~mask, 0.)
    total = r.sum(1)
    valid = total > 0
    q = r / total.clamp_min(1e-12).unsqueeze(1)
    logp = torch.log_softmax(lg, 1).masked_fill(~mask, 0.)
    return -(q * logp).sum(1), valid


def losses(torch, logits, rate, idx, mask):
    """(total, mean soft BCE over rows, mean listwise CE over valid decisions, n valid)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, rate, reduction='mean')
    terms, valid = listwise_terms(torch, logits, rate, idx, mask)
    nv = int(valid.sum())
    lw = terms[valid].sum() / max(1, nv)
    return bce + LISTWISE_W * lw, bce, lw, nv


# -------------------------------------------------------------------- train
def device_of(torch, name):
    return ab.device_of(torch, name)


def batches_of(dec_order, per_batch):
    return [dec_order[i:i + per_batch] for i in range(0, len(dec_order), per_batch)]


def gather_rows(data, decs):
    """Row ids of the decisions (in order) and the batch-relative decision starts."""
    sizes = data['start'][decs + 1] - data['start'][decs]
    rows = np.concatenate([np.arange(data['start'][d], data['start'][d + 1]) for d in decs])
    return rows, np.concatenate([[0], np.cumsum(sizes)])


def score(torch, net, data, decs, mean, sd, device):
    """Logits (float64) of all rows of ``decs`` (no augmentation, eval mode), in row order."""
    net.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(decs), max(1, EVAL_BATCH // N_CANDIDATES)):
            rows, _ = gather_rows(data, decs[lo:lo + EVAL_BATCH // N_CANDIDATES])
            x = (torch.from_numpy(data['x'][rows]).to(device) - mean) / sd
            e = torch.from_numpy(data['extra'][rows]).to(device)
            out.append(net(x, e).cpu().double().numpy())
    return np.concatenate(out)


def val_losses(torch, net, data, decs, mean, sd, device):
    logits = torch.from_numpy(score(torch, net, data, decs, mean, sd, device))
    rows, starts = gather_rows(data, decs)
    rate = torch.from_numpy(data['rate'][rows])
    idx, mask = decision_index(torch, starts, 'cpu')
    _, bce, lw, nv = losses(torch, logits, rate, idx, mask)
    return float(bce), float(lw), nv


def train(args, cfg):
    import torch
    m = cfg['model']
    epochs = args.epochs or m['epochs']
    torch.manual_seed(m['seed'])
    device = device_of(torch, args.device)
    board = layout()
    data = load(args.data, cfg, 'train', args.max_games)
    lo, n = split_ranges(cfg)['train']
    cut = lo + n - validation_games(cfg)
    fit_decs = np.flatnonzero(data['dec_game'] < cut)
    val_decs = np.flatnonzero(data['dec_game'] >= cut)
    if not len(fit_decs) or not len(val_decs):
        raise ValueError('empty fit or validation decisions')
    fit_rows = gather_rows(data, fit_decs)[0]
    perms_cpu = symmetry_index(torch)
    mean, sd = standardization(torch, data['x'], fit_rows, perms_cpu)
    perms, dmean, dsd = perms_cpu.to(device), mean.to(device), sd.to(device)
    net = make_model(torch, board).to(device)
    params = sum(p.numel() for p in net.parameters())
    per_batch = max(1, m['batch'] // N_CANDIDATES)
    steps = epochs * math.ceil(len(fit_decs) / per_batch)
    opt, sched = make_optimizer(torch, net, m, steps)
    gen = torch.Generator().manual_seed(m['seed'])
    rate_all = torch.from_numpy(data['rate'].astype(np.float32))
    info = {'device': str(device), 'params': params, 'fit_decisions': len(fit_decs), 'val_decisions': len(val_decs),
            'fit_rows': len(fit_rows), 'decisions_per_batch': per_batch, 'steps': steps, 'epochs': epochs}
    print(json.dumps(info), flush=True)
    best, history, started = None, [], time.monotonic()
    for epoch in range(epochs):
        net.train()
        t0 = time.monotonic()
        sums = {'loss': 0., 'bce': 0., 'listwise': 0., 'rows': 0, 'listwise_decisions': 0}
        order = fit_decs[torch.randperm(len(fit_decs), generator=gen).numpy()]
        for decs in batches_of(order, per_batch):
            rows, starts = gather_rows(data, decs)
            sizes = np.diff(starts)
            # one symmetry per decision, shared by its candidates
            pick = torch.randint(0, len(perms_cpu), (len(decs),), generator=gen)
            pick = torch.repeat_interleave(pick, torch.from_numpy(sizes)).to(device)
            xb = (torch.from_numpy(data['x'][rows]).to(device) - dmean) / dsd
            xb = torch.gather(xb, 1, perms[pick])
            eb = torch.from_numpy(data['extra'][rows]).to(device)
            rb = rate_all[torch.from_numpy(rows)].to(device)
            idx, mask = decision_index(torch, starts, device)
            loss, bce, lw, nv = losses(torch, net(xb, eb), rb, idx, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            sums['loss'] += loss.item() * len(rows)
            sums['bce'] += bce.item() * len(rows)
            sums['listwise'] += lw.item() * nv
            sums['rows'] += len(rows)
            sums['listwise_decisions'] += nv
        vb, vl, vn = val_losses(torch, net, data, val_decs, dmean, dsd, device)
        entry = {'epoch': epoch, 'train_loss': sums['loss'] / sums['rows'], 'train_bce': sums['bce'] / sums['rows'],
                 'train_listwise': sums['listwise'] / max(1, sums['listwise_decisions']),
                 'val_bce': vb, 'val_listwise': vl, 'val_listwise_decisions': vn,
                 'epoch_seconds': time.monotonic() - t0, 'seconds': time.monotonic() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if best is None or vl < best[0]:
            best = (vl, epoch, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    seconds = time.monotonic() - started
    smoke = bool(args.max_games or args.epochs)
    path = args.out / 'model.pt'
    publish(path, lambda tmp: torch.save({
        'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1], 'val_listwise': best[0], 'config': cfg,
        'train_identity': data['identity'], 'train_seeds': data['seeds'], 'params': params,
        'fit_mean_rate': float(data['rate'][fit_rows].mean()),
        'train_seconds': seconds, 'smoke': smoke, 'max_games': args.max_games, 'epochs': epochs}, tmp))
    return {**info, 'fit_games': int(len(np.unique(data['dec_game'][fit_decs]))),
            'val_games': int(len(np.unique(data['dec_game'][val_decs]))), 'rows': len(data['rate']),
            'decisions': len(data['start']) - 1, 'candidate_counts': data['candidate_counts'],
            'chosen_epoch': best[1], 'chosen_val_listwise': best[0], 'history': history,
            'train_seconds': seconds, 'max_games': args.max_games, 'smoke': smoke,
            'train_identity': data['identity'], 'train_file_sha256': {str(k): v for k, v in data['hashes'].items()},
            'model_sha256': sha256(path)}


# --------------------------------------------------------------------- gate
def gate_constants(cfg):
    text = cfg['gate']['primary']
    m = re.search(r'lower bound > (\d+(?:\.\d+)?) \((\d[\d,]*) resamples, seed (\d+)\)', text)
    if not m:
        raise ValueError('cannot read the gate constants from gate.primary')
    return {'lcb_gt': float(m.group(1)), 'resamples': int(m.group(2).replace(',', '')), 'seed': int(m.group(3))}


def bootstrap(values, groups, resamples, seed):
    """Game-clustered bootstrap of the decision mean: (point, [lcb, ucb]) with the
    ``arch_bakeoff`` percentile convention (sorted[int(.025 B)], sorted[int(.975 B) - 1])."""
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(int(g), []).append(i)
    ids = sorted(by)
    by = {g: np.asarray(v) for g, v in by.items()}
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        sample = np.concatenate([by[rng.choice(ids)] for _ in ids])
        stats.append(float(values[sample].mean()))
    stats.sort()
    return float(values.mean()), [stats[int(.025 * len(stats))], stats[int(.975 * len(stats)) - 1]]


def midranks(v):
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(a, b):
    """Spearman with midranks; None when either side is constant."""
    ra, rb = midranks(np.asarray(a, dtype=np.float64)), midranks(np.asarray(b, dtype=np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def playout_auc(scores, wins, playouts):
    """AUC of row scores against the individual playout outcomes (wins positives,
    playouts - wins negatives per row)."""
    s = np.concatenate([np.repeat(scores, wins), np.repeat(scores, playouts - wins)])
    y = np.concatenate([np.ones(int(wins.sum())), np.zeros(int((playouts - wins).sum()))])
    return fast_auc(s, y)


def soft_log_loss(p, rate):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(rate * np.log(p) + (1 - rate) * np.log(1 - p)))


def decision_stats(data, scores):
    """Per decision: pick (top score; first on ties), its rate, served rate, mean rate, max rate,
    unique-best flags and Spearman(scores, rates)."""
    out = {k: [] for k in ('pick', 'served', 'mean', 'best', 'pick_unique_best', 'pick_tied_best',
                           'served_unique_best', 'served_tied_best', 'random_unique_best', 'random_tied_best',
                           'spearman', 'spearman_served')}
    for d in range(len(data['start']) - 1):
        a, b = data['start'][d], data['start'][d + 1]
        r, s, rank = data['rate'][a:b], scores[a:b], data['rank'][a:b]
        best = r.max()
        unique = int((r == best).sum()) == 1
        p, sv = int(np.argmax(s)), int(np.flatnonzero(rank == 0)[0])
        out['pick'].append(r[p])
        out['served'].append(r[sv])
        out['mean'].append(r.mean())
        out['best'].append(best)
        out['pick_unique_best'].append(float(unique and r[p] == best))
        out['pick_tied_best'].append(float(r[p] == best))
        out['served_unique_best'].append(float(unique and r[sv] == best))
        out['served_tied_best'].append(float(r[sv] == best))
        out['random_unique_best'].append(float(unique) / len(r))
        out['random_tied_best'].append(float((r == best).sum()) / len(r))
        out['spearman'].append(spearman(s, r))
        out['spearman_served'].append(spearman(-rank, r))
    return {k: (np.asarray(v, dtype=np.float64) if not k.startswith('spearman') else v) for k, v in out.items()}


def mean_defined(values):
    v = [x for x in values if x is not None]
    return (float(np.mean(v)) if v else None), len(v)


def gate_report(data, logits, consts, train_mean_rate):
    p = 1 / (1 + np.exp(-logits))
    st = decision_stats(data, logits)
    groups = data['dec_game']
    diff = st['pick'] - st['served']
    point, ci = bootstrap(diff, groups, consts['resamples'], consts['seed'])
    rnd_point, rnd_ci = bootstrap(st['mean'] - st['served'], groups, consts['resamples'], consts['seed'])

    def by(key_arr, labels):
        return {lab: {'pick_minus_served': float(diff[key_arr == k].mean()), 'decisions': int((key_arr == k).sum())}
                for k, lab in labels if (key_arr == k).any()}
    first = (data['dec_setup'] >= N_SETUP // 2).astype(int)
    sp, sp_n = mean_defined(st['spearman'])
    sps, sps_n = mean_defined(st['spearman_served'])
    rate, wins, playouts = data['rate'], data['wins'], data['playouts']
    return {
        'primary': {'pick_minus_served': point, 'ci95': ci, 'lcb': ci[0], 'pass': bool(ci[0] > consts['lcb_gt']),
                    'decisions': len(diff), 'games': int(len(np.unique(groups)))},
        'rates': {'model_pick': float(st['pick'].mean()), 'served': float(st['served'].mean()),
                  'random': float(st['mean'].mean()), 'in_sample_best': float(st['best'].mean())},
        'by_setup_index': by(data['dec_setup'], [(k, str(k)) for k in range(N_SETUP)]),
        'by_placement': by(first, [(0, 'first'), (1, 'second')]),
        'by_seat': by(data['dec_seat'], [(k, str(k)) for k in range(4)]),
        'model': {'regret_vs_in_sample_best': float((st['best'] - st['pick']).mean()),
                  'top1_unique_best': float(st['pick_unique_best'].mean()),
                  'top1_tied_best': float(st['pick_tied_best'].mean()),
                  'spearman_mean': sp, 'spearman_decisions': sp_n,
                  'row_log_loss': soft_log_loss(p, rate), 'row_auc': playout_auc(logits, wins, playouts)},
        'baselines': {
            'served': {'pick_minus_served': 0., 'regret_vs_in_sample_best': float((st['best'] - st['served']).mean()),
                       'top1_unique_best': float(st['served_unique_best'].mean()),
                       'top1_tied_best': float(st['served_tied_best'].mean()),
                       'spearman_mean': sps, 'spearman_decisions': sps_n,
                       'row_log_loss': None, 'row_auc': playout_auc(-data['rank'].astype(np.float64), wins, playouts),
                       'note': 'scores = -served_rank; no probability, so no log loss'},
            'random': {'pick_minus_served': rnd_point, 'ci95': rnd_ci,
                       'regret_vs_in_sample_best': float((st['best'] - st['mean']).mean()),
                       'top1_unique_best': float(st['random_unique_best'].mean()),
                       'top1_tied_best': float(st['random_tied_best'].mean()),
                       'spearman_mean': 0., 'row_auc': .5,
                       'row_log_loss': soft_log_loss(np.full(len(rate), train_mean_rate), rate),
                       'note': 'expected values of a uniform pick; spearman 0 and AUC .5 by symmetry; log loss of '
                               'the constant train-fit mean rate'}},
        'notes': {'top1_unique_best': 'pick is the unique in-sample playout-best candidate (the 11.8% statistic)',
                  'regret': 'descriptive only: 8-playout selection noise biases it',
                  'row_auc': 'against individual playout outcomes'}}


def gate(args, cfg):
    import torch
    device = device_of(torch, args.device)
    board = layout()
    path = args.out / 'model.pt'
    saved = torch.load(path, weights_only=False, map_location='cpu')
    if saved['config'] != cfg:
        raise ValueError('checkpoint was trained under a different declaration')
    if saved['smoke'] and not args.allow_smoke:
        raise ValueError('refusing a smoke checkpoint (pass --allow-smoke for a smoke gate)')
    tlo, tn = split_ranges(cfg)['test']
    if set(saved['train_seeds']) & set(range(tlo, tlo + tn)):
        raise ValueError('checkpoint train seeds overlap the test split')
    train_seeds = selected_seeds(cfg, 'train', saved['max_games'])
    if saved['train_seeds'] != train_seeds or identity(file_hashes(args.data, train_seeds)) != saved['train_identity']:
        raise ValueError('train data differs from the checkpoint train identity')
    data = load(args.data, cfg, 'test')
    net = make_model(torch, board)
    net.load_state_dict(saved['state'])
    net.to(device)
    decs = np.arange(len(data['start']) - 1)
    logits = score(torch, net, data, decs, saved['mean'].to(device), saved['sd'].to(device), device)
    consts = gate_constants(cfg)
    report = gate_report(data, logits, consts, saved['fit_mean_rate'])
    return {**report, 'bar': consts, 'smoke': bool(saved['smoke']), 'allow_smoke': bool(args.allow_smoke),
            'device': str(device), 'test_rows': len(data['rate']), 'candidate_counts': data['candidate_counts'],
            'test_identity': data['identity'], 'test_file_sha256': {str(k): v for k, v in data['hashes'].items()},
            'train_identity': saved['train_identity'], 'model_sha256': sha256(path), 'params': saved['params'],
            'chosen_epoch': saved['epoch'], 'chosen_val_listwise': saved['val_listwise'],
            'train_seconds': saved['train_seconds'], 'epochs': saved['epochs'], 'max_games': saved['max_games']}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'gate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='placement-data-v1 root (config.json, games/)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto', help='auto (mps if available, else cpu), cpu or mps')
    parser.add_argument('--max-games', type=int, default=None, help='smoke only (train)')
    parser.add_argument('--epochs', type=int, default=None, help='smoke only (train)')
    parser.add_argument('--allow-smoke', action='store_true', help='gate a smoke checkpoint')
    args = parser.parse_args(argv)
    if args.command == 'gate' and (args.epochs or args.max_games):
        parser.error('--epochs/--max-games apply to train only')
    cfg = json.loads(args.config.read_text())
    made = json.loads((args.data / 'config.json').read_text())['config']
    if {k: list(v) for k, v in made['split'].items()} != cfg['data']['split']:
        raise ValueError('data split differs from the declared split')
    check_model_config(cfg)
    gate_constants(cfg)
    validation_games(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / ('train.json' if args.command == 'train' else 'gate.json')
    refuse_existing(output, *((args.out / 'model.pt',) if args.command == 'train' else ()))
    started = time.monotonic()
    result = (train if args.command == 'train' else gate)(args, cfg)
    result.update({'command': args.command, 'config_sha256': sha256(args.config),
                   'script_sha256': sha256(Path(__file__)), 'data_config_sha256': sha256(args.data / 'config.json'),
                   'seconds': time.monotonic() - started})
    publish(output, lambda tmp: tmp.write_text(json.dumps(result, indent=1)))
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'train_file_sha256',
                                                                    'test_file_sha256')}, indent=1))
    return result


if __name__ == '__main__':
    main()
