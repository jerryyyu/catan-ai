"""tiles_margin arm (experiments/tiles-margin.json): the arch-bakeoff ``tiles`` transformer
with an auxiliary final-VP margin head.

Model: ``arch_bakeoff.make_net(torch, 'tiles')`` unchanged (trunk and P(I win) head, kept
as ``trunk.*``) plus ``margin``, a linear head on the final global-token representation
(the LayerNorm output the win head reads). Loss per batch: mean BCE of P(I win) over the
rows + 0.5 x mean Huber(delta 1) of the margin over the rows that have a margin target.
Everything else (verified shard loader with array binding, symmetrized standardization,
12-symmetry augmentation, AdamW/cosine schedule, epoch selection by validation BCE of
P(I win) on the last 5,000 train games) is ``arch_bakeoff``'s, with the common settings
read from experiments/arch-bakeoff.json (``--base-config``) and checked against the
values stated in this arm's declaration.

Margin target per row: seat = argmax of the encoder-v2 absolute-seat one-hot (as the
``heads`` arm); margin = (seat's true final VP - max true final VP of the other three
seats) / 10, from ``scripts/replay_final_points.py`` outputs (``--final-points`` holds one
directory per replayed root, each with ``manifest.json`` and ``games/<seed>.json``). Every
final-points file must carry the SHA-256 of the game record the shards were made from.
Rows of games whose replay did not match (``match: false``) or that were truncated are
excluded from the margin loss and counted; they still train the win head.

Commands (``--max-rows`` / ``--epochs`` are smoke overrides, marked ``smoke: true``):

    tiles_margin.py train --config experiments/tiles-margin.json \
        --base-config experiments/arch-bakeoff.json --data runs/value-v2-100k-shards \
        --final-points runs/value100k-final-points-v1 --out runs/tiles-margin-v1
    tiles_margin.py win --config ... --out runs/tiles-margin-v1
        # tiles_margin-win.pt: the trunk + win head as a plain arch-bakeoff 'tiles'
        # checkpoint (margin head dropped) for cross_eval_arch.py and export_tiles_npz.py
    tiles_margin.py gate --config ... --base-config ... --data ... --final-points ... \
        --out runs/tiles-margin-v1 --reference runs/arch-bakeoff-v1/tiles.pt \
        --xeval runs/tiles-margin-v1/xeval.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_bakeoff as ab  # noqa: E402
from artifact_guard import publish, refuse_existing, sha256  # noqa: E402
from train_value import auc  # noqa: E402
from train_value_v2 import make_optimizer, shards, symmetry_index  # noqa: E402

ARM = 'tiles_margin'
MARGIN_SCALE = 10.


# ------------------------------------------------------------------ config
def margin_constants(cfg):
    """(loss weight, Huber delta) as written in ``arm.loss``."""
    m = re.search(r'BCE\(P\(I win\)\) \+ ([\d.]+) x Huber\(margin/10, delta ([\d.]+)\)', cfg['arm']['loss'])
    if not m:
        raise ValueError('cannot read the margin loss from arm.loss')
    return float(m.group(1)), float(m.group(2))


def gate_constants(cfg):
    """The offline gate as written in ``offline_gate.primary``."""
    text = cfg['offline_gate']['primary'].replace('−', '-')
    m = re.search(r'AUC\(arch-bakeoff-v1 tiles, ([\d.]+)\) >= (-?[\d.]+) with the game-clustered bootstrap 95% '
                  r'LCB > (-?[\d.]+) \(([\d,]+) resamples, seed (\d+)\).*>= the tiles model\'s \+([\d.]+)', text)
    if not m:
        raise ValueError('cannot read the gate constants from offline_gate.primary')
    return {'reference_auc': float(m.group(1)), 'min_diff': float(m.group(2)), 'lcb_gt': float(m.group(3)),
            'resamples': int(m.group(4).replace(',', '')), 'seed': int(m.group(5)),
            'xeval_min': float(m.group(6))}


def check_configs(cfg, base):
    """The base declaration must be the arch-bakeoff one this arm names, with the stated
    common settings (AdamW 7e-4, wd .05, cosine 3% warmup, 12 epochs, batch 4096, seed 0)."""
    if cfg.get('name') != 'tiles-margin' or base.get('name') != 'arch-bakeoff':
        raise ValueError('expected experiments/tiles-margin.json and experiments/arch-bakeoff.json')
    want = {'optimizer': 'adamw', 'learning_rate': 0.0007, 'weight_decay': 0.05, 'schedule': 'cosine',
            'warmup_fraction': 0.03, 'epochs': 12, 'batch': 4096, 'validation_games': 5000, 'seed': 0}
    got = {k: base['common'].get(k) for k in want}
    if got != want:
        raise ValueError(f'arch-bakeoff common settings differ from the declared ones: {got}')
    if (ab.TILES_D, ab.TILES_LAYERS, ab.TILES_HEADS, ab.TILES_MLP, ab.TILES_DROPOUT) != (128, 4, 4, 256, .1):
        raise ValueError('arch_bakeoff tiles constants differ from the declared model')
    margin_constants(cfg)
    gate_constants(cfg)


# ------------------------------------------------------------------ model
def make_margin_net(torch, board=None):
    nn = torch.nn

    class TilesMargin(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = ab.make_net(torch, 'tiles', board)
            self.margin = nn.Linear(ab.TILES_D, 1)

        def forward(self, x):
            t, n = self.trunk, len(x)
            h = torch.cat([t.tile(x[:, t.tidx]), t.player(x[:, ab.P0:ab.G0].reshape(n, ab.N_PLAYERS, ab.PLAYER_W)),
                           t.glob(x[:, ab.G0:]).unsqueeze(1)], 1) + t.pos + t.kind[t.types]
            g = t.norm(t.encoder(h)[:, -1])
            return t.head(g).squeeze(-1), self.margin(g).squeeze(-1)

    return TilesMargin()


def win_state(state):
    """The plain ``tiles`` state dict (trunk + win head) of a tiles_margin state dict."""
    keys = set(state)
    if not keys or any(not (k.startswith('trunk.') or k.startswith('margin.')) for k in keys):
        raise ValueError('not a tiles_margin state dict')
    return {k[len('trunk.'):]: v for k, v in state.items() if k.startswith('trunk.')}


def batch_loss(torch, win_logit, margin_pred, y, target, mask, weight, delta):
    """(loss, bce_sum, huber_sum, margin rows): mean BCE over all rows + weight x mean
    Huber over the rows with a margin target (``mask``)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(win_logit, y, reduction='sum')
    k = int(mask.sum())
    hub = (torch.nn.functional.huber_loss(margin_pred[mask], target[mask], reduction='sum', delta=delta)
           if k else win_logit.new_zeros(()))
    return bce / len(y) + weight * hub / max(1, k), bce, hub, k


# ------------------------------------------------------------------ targets
def load_final_points(root, seeds, sources):
    """({seed: final-points record}, replay manifests, digest) for ``seeds``, each record bound
    to the game record the shards were made from (``record_sha256`` == the shard manifest's
    source hash). The digest is the SHA-256 over ``seed:sha256(file bytes)`` lines in seed
    order: it binds the exact replay records consumed (points included), not only their
    producer metadata (Codex HOLD @b426ea8, P2)."""
    dirs = sorted(p.parent for p in Path(root).glob('*/manifest.json'))
    if not dirs:
        raise ValueError(f'no replay directories (*/manifest.json) under {root}')
    manifests = {d.name: json.loads((d / 'manifest.json').read_text()) for d in dirs}
    out, lines = {}, []
    for s in sorted(int(v) for v in seeds):
        hits = [d for d in dirs if (d / 'games' / f'{s}.json').exists()]
        if len(hits) != 1:
            raise ValueError(f'seed {s}: {len(hits)} final-points records (need exactly one)')
        raw = (hits[0] / 'games' / f'{s}.json').read_bytes()
        lines.append(f'{s}:{hashlib.sha256(raw).hexdigest()}\n')
        rec = json.loads(raw)
        if rec['seed'] != s or rec['manifest_sha256'] != manifests[hits[0].name]['manifest_sha256']:
            raise ValueError(f'final points for seed {s} do not belong to their replay manifest')
        if rec['record_sha256'] != sources[s]:
            raise ValueError(f'final points for seed {s} were replayed from a different game record')
        out[s] = rec
    binding = {name: {k: m.get(k) for k in ('manifest_sha256', 'producer_commit', 'root_identity_sha256',
                                             'script_sha256', 'engine_backend', 'source_drift')}
               for name, m in manifests.items()}
    return out, binding, hashlib.sha256(''.join(lines).encode()).hexdigest()


def targets_digest(target, mask, games):
    """SHA-256 of the supervised margin arrays (row order of the verified shards)."""
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(target, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.bool_),
              np.ascontiguousarray(games, dtype=np.int64)):
        h.update(a.tobytes())
    return h.hexdigest()


def margin_binding(final_points, data, sources):
    """Replay-record and target-array digests for ``data``'s rows (train or test)."""
    final, manifests, records = load_final_points(final_points, np.unique(data['game']), sources)
    target, mask, counts = margin_targets(data, final)
    return target, mask, counts, {'manifests': manifests, 'records_sha256': records,
                                  'targets_sha256': targets_digest(target, mask, data['game'])}


def margin_targets(data, final):
    """Per row: margin / 10 (float32) and the mask of rows that have a margin target;
    checks winner/label consistency on matched rows."""
    x = data['x']
    onehot = x[:, ab.SEAT_COLS].astype(np.float32)
    if not np.all(onehot.sum(1) == 1):
        raise ValueError('absolute-seat one-hot is not one-hot on every row')
    seat = onehot.argmax(1)
    games, inverse = np.unique(data['game'], return_inverse=True)
    pts = np.array([final[int(g)]['points'] for g in games], dtype=np.float32)
    ok = np.array([bool(final[int(g)]['match']) and final[int(g)]['winner'] is not None for g in games])
    winner = np.array([-1 if final[int(g)]['winner'] is None else final[int(g)]['winner'] for g in games])
    rp = pts[inverse]
    mine = rp[np.arange(len(seat)), seat]
    others = np.where(np.arange(4)[None, :] == seat[:, None], -np.inf, rp).max(1)
    target = ((mine - others) / MARGIN_SCALE).astype(np.float32)
    mask = ok[inverse]
    y = data['y'].astype(int)
    bad = int(np.sum((winner[inverse][mask] == seat[mask]) != (y[mask] == 1)))
    if bad:
        raise ValueError(f'winner/label consistency check failed on {bad} matched rows')
    mismatched = [int(g) for g, f in zip(games, ok) if not f and final[int(g)]['winner'] is not None]
    counts = {'rows': len(y), 'margin_rows': int(mask.sum()), 'excluded_rows': int((~mask).sum()),
              'games': len(games), 'mismatch_games': len(mismatched), 'mismatch_seeds': mismatched[:50],
              'truncated_games': int(sum(final[int(g)]['winner'] is None for g in games)),
              'won_rows_with_nonpositive_margin': int(np.sum((y == 1) & mask & (target <= 0))),
              'target_mean': float(target[mask].mean()) if mask.any() else None,
              'target_sd': float(target[mask].std()) if mask.any() else None}
    return np.where(mask, target, 0.).astype(np.float32), mask, counts


# ------------------------------------------------------------------ scoring
def score_margin(torch, net, x, rows, mean, sd, device):
    """P(I win) (float64) and the margin prediction (float64), without augmentation."""
    net.eval()
    p, m = [], []
    with torch.no_grad():
        for lo in range(0, len(rows), ab.EVAL_BATCH):
            w, mg = net(ab.to_batch(torch, x, rows[lo:lo + ab.EVAL_BATCH], mean, sd, device))
            p.append(torch.sigmoid(w).cpu().double().numpy())
            m.append(mg.cpu().double().numpy())
    return np.concatenate(p), np.concatenate(m)


def margin_report(pred, target, mask):
    if not mask.any():
        return {'rows': 0}
    a, b = pred[mask], target[mask].astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {'rows': int(mask.sum()), 'mae_vp': float(np.abs(a - b).mean() * MARGIN_SCALE), 'corr': corr}


# ------------------------------------------------------------------ train
def train(args, cfg, base):
    import torch
    common = base['common']
    weight, delta = margin_constants(cfg)
    epochs = args.epochs or common['epochs']
    torch.manual_seed(common['seed'])
    device = ab.device_of(torch, args.device)
    board = ab.layout()
    verified = shards(args.data, 'train')
    data = ab.subsample(verified, args.max_rows, common['seed'])
    sources = ab.manifests(args.data, 'train')[1]
    lo, n = ab.declared_split(base)['train']
    if data['game'].min() < lo or data['game'].max() >= lo + n:
        raise ValueError('train shards hold games outside the declared train split')
    cut = lo + n - common['validation_games']
    fit_rows = np.flatnonzero(data['game'] < cut)
    val_rows = np.flatnonzero(data['game'] >= cut)
    if not len(fit_rows) or not len(val_rows):
        raise ValueError('empty fit or validation rows')
    target_np, mask_np, counts, fp_binding = margin_binding(args.final_points, data, sources)
    x, y = data['x'], torch.from_numpy(data['y'].astype(np.float32))
    target, mask = torch.from_numpy(target_np), torch.from_numpy(mask_np)
    perms_cpu = symmetry_index(torch)
    mean, sd = ab.standardization(torch, x, fit_rows, perms_cpu)
    perms, dmean, dsd = perms_cpu.to(device), mean.to(device), sd.to(device)
    net = make_margin_net(torch, board).to(device)
    params = sum(p.numel() for p in net.parameters())
    batch = common['batch']
    steps = epochs * ((len(fit_rows) + batch - 1) // batch)
    opt, sched = make_optimizer(torch, net, common, steps)
    gen = torch.Generator().manual_seed(common['seed'])
    yv = data['y'][val_rows].astype(np.float64)
    best, history, started = None, [], time.monotonic()
    print(json.dumps({'arm': ARM, 'device': str(device), 'params': params, 'fit_rows': len(fit_rows),
                      'val_rows': len(val_rows), 'epochs': epochs, 'counts': counts}), flush=True)
    for epoch in range(epochs):
        net.train()
        sums = {'loss': 0., 'bce': 0., 'huber': 0., 'margin_rows': 0}
        for idx in torch.randperm(len(fit_rows), generator=gen).split(batch):
            rows = np.sort(fit_rows[idx.numpy()])
            xb = ab.to_batch(torch, x, rows, dmean, dsd, device)
            pick = torch.randint(0, len(perms_cpu), (len(rows),), generator=gen).to(device)
            xb = torch.gather(xb, 1, perms[pick])
            rt = torch.from_numpy(rows)
            w, mg = net(xb)
            loss, bce, hub, k = batch_loss(torch, w, mg, y[rt].to(device), target[rt].to(device),
                                           mask[rt].to(device), weight, delta)
            sums['loss'] += loss.item() * len(rows)
            sums['bce'] += bce.item()
            sums['huber'] += hub.item()
            sums['margin_rows'] += k
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        p, mp = score_margin(torch, net, x, val_rows, dmean, dsd, device)
        val_bce = ab.log_loss(p, yv)
        entry = {'epoch': epoch, 'train_loss': sums['loss'] / len(fit_rows), 'train_bce': sums['bce'] / len(fit_rows),
                 'train_huber': sums['huber'] / max(1, sums['margin_rows']), 'val_bce': val_bce,
                 'val_auc': ab.fast_auc(p, yv), 'val_margin': margin_report(mp, target_np[val_rows], mask_np[val_rows]),
                 'seconds': time.monotonic() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if best is None or val_bce < best[0]:
            best = (val_bce, epoch, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    seconds = time.monotonic() - started
    smoke = bool(args.max_rows or args.epochs)
    path = args.out / f'{ARM}.pt'
    publish(path, lambda tmp: torch.save({
        'arm': ARM, 'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1], 'val_bce': best[0], 'config': cfg,
        'base_config': base, 'margin_weight': weight, 'huber_delta': delta, 'margin_counts': counts,
        'final_points': fp_binding, 'train_identity': verified['identity'],
        'train_arrays_sha256': verified['arrays_sha256'],
        'train_sealed_after_the_fact': verified['sealed_after_the_fact'], 'params': params,
        'train_seconds': seconds, 'smoke': smoke, 'max_rows': args.max_rows, 'epochs': epochs}, tmp))
    return {'arm': ARM, 'fit_rows': len(fit_rows), 'val_rows': len(val_rows),
            'fit_games': int(len(np.unique(data['game'][fit_rows]))),
            'val_games': int(len(np.unique(data['game'][val_rows]))),
            'chosen_epoch': best[1], 'chosen_val_bce': best[0], 'history': history, 'params': params,
            'train_seconds': seconds, 'device': str(device), 'epochs': epochs, 'max_rows': args.max_rows,
            'smoke': smoke, 'margin_counts': counts, 'final_points': fp_binding,
            'train_identity': verified['identity'], 'train_arrays_sha256': verified['arrays_sha256'],
            'train_sealed_after_the_fact': verified['sealed_after_the_fact'], 'model_sha256': sha256(path)}


# ------------------------------------------------------------------ win-head checkpoint
def win_checkpoint(args, cfg, base):
    """Write ``tiles_margin-win.pt``: an arch-bakeoff 'tiles' checkpoint (trunk + win head;
    the margin head dropped) read unchanged by cross_eval_arch.py and export_tiles_npz.py."""
    import torch
    src = args.out / f'{ARM}.pt'
    saved = torch.load(src, weights_only=False, map_location='cpu')
    if saved.get('arm') != ARM or saved['config'] != cfg:
        raise ValueError('not a tiles_margin checkpoint of this declaration')
    state = win_state(saved['state'])
    net = ab.make_net(torch, 'tiles', ab.layout())
    net.load_state_dict(state)  # strict: exactly the tiles keys
    path = args.out / f'{ARM}-win.pt'
    keep = ('mean', 'sd', 'epoch', 'val_bce', 'config', 'train_identity', 'train_arrays_sha256',
            'train_sealed_after_the_fact', 'train_seconds', 'smoke', 'max_rows', 'epochs')
    publish(path, lambda tmp: torch.save({
        **{k: saved[k] for k in keep}, 'arm': 'tiles', 'state': state,
        'params': sum(p.numel() for p in net.parameters()),
        'derived_from': {'arm': ARM, 'pt_sha256': sha256(src), 'params_with_margin': saved['params'],
                         'final_points_records_sha256': saved['final_points']['records_sha256'],
                         'margin_targets_sha256': saved['final_points']['targets_sha256'],
                         'dropped': sorted(k for k in saved['state'] if k.startswith('margin.'))}}, tmp))
    return {'win_pt': str(path), 'win_sha256': sha256(path), 'source_sha256': sha256(src), 'smoke': saved['smoke']}


# ------------------------------------------------------------------ gate
def xeval_check(xeval, bar):
    """Reasons the cross-eval receipt fails (empty: it passes). Fail closed (Codex HOLD @b426ea8,
    P1): besides the declared point bar, ``cross_eval_arch``'s own eligibility must be
    explicitly true, i.e. its 95% LCB > 0 and rollout truncation recorded and zero; a
    missing field fails."""
    out = []
    point = xeval.get('pick_minus_served')
    if not isinstance(point, (int, float)) or not point >= bar:
        out.append(f'pick_minus_served {point} < declared bar {bar}')
    if xeval.get('eligible_for_next_gate') is not True:
        out.append(f"cross_eval_arch eligibility not true: {xeval.get('ineligible_reason', 'field missing')}")
    lcb = xeval.get('pick_minus_served_lcb95')
    if not isinstance(lcb, (int, float)) or not lcb > 0:
        out.append(f'pick_minus_served_lcb95 {lcb} is not > 0')
    trunc = xeval.get('truncation') or {}
    if trunc.get('violation') is not None or trunc.get('rollout_truncations') != 0:
        out.append(f"rollout truncation not recorded as zero: {trunc.get('violation', 'truncation missing')}")
    return out


def gate(args, cfg, base):
    import torch
    g = gate_constants(cfg)
    device = ab.device_of(torch, args.device)
    board = ab.layout()
    path = args.out / f'{ARM}.pt'
    saved = torch.load(path, weights_only=False, map_location='cpu')
    trained = shards(args.data, 'train')
    train_binding = {k: trained[k] for k in ('identity', 'arrays_sha256', 'sealed_after_the_fact')}
    # the replay records and margin arrays the checkpoint was trained on (Codex HOLD @b426ea8, P2)
    train_fp = margin_binding(args.final_points, ab.subsample(trained, saved['max_rows'], base['common']['seed']),
                              ab.manifests(args.data, 'train')[1])[3]
    del trained
    for key in ('records_sha256', 'targets_sha256'):
        if saved.get('final_points', {}).get(key) != train_fp[key]:
            raise ValueError(f'tiles_margin checkpoint was trained on different replay targets ({key})')
    test = ab.subsample(shards(args.data, 'test'), args.max_rows, base['common']['seed'])
    lo, n = ab.declared_split(base)['test']
    if test['game'].min() < lo or test['game'].max() >= lo + n:
        raise ValueError('test shards hold games outside the declared test split')
    rows = np.arange(len(test['y']))
    y = test['y'].astype(np.float64)
    if (saved['arm'] != ARM or saved['config'] != cfg or saved['train_identity'] != train_binding['identity']
            or saved['train_arrays_sha256'] != train_binding['arrays_sha256']):
        raise ValueError('tiles_margin checkpoint provenance does not match this declaration and training data')
    ref = torch.load(args.reference, weights_only=False, map_location='cpu')
    if ref.get('arm') != 'tiles' or ref['train_identity'] != train_binding['identity']:
        raise ValueError('reference is not a tiles checkpoint trained on these shards')
    if 'train_arrays_sha256' in ref and ref['train_arrays_sha256'] != train_binding['arrays_sha256']:
        raise ValueError('reference was trained on different shard arrays')
    net = make_margin_net(torch, board)
    net.load_state_dict(saved['state'])
    net.to(device)
    p, mp = score_margin(torch, net, test['x'], rows, saved['mean'].to(device), saved['sd'].to(device), device)
    rnet = ab.make_net(torch, 'tiles', board)
    rnet.load_state_dict(ref['state'])
    rnet.to(device)
    pr, _ = ab.score(torch, rnet, 'tiles', test['x'], rows, ref['mean'].to(device), ref['sd'].to(device), device)
    smoke = bool(args.max_rows or saved['smoke'] or ref.get('smoke'))
    result = {}
    for name, pred, ck, cpath in ((ARM, p, saved, path), ('tiles_reference', pr, ref, args.reference)):
        point = ab.fast_auc(pred, y)
        if abs(point - auc(pred.tolist(), y.astype(int).tolist())) > 1e-9:
            raise ValueError('fast AUC disagrees with train_value.auc')
        result[name] = {'auc': point, 'log_loss': ab.log_loss(pred, y), 'ece10': ab.ece(pred, y),
                        'params': ck['params'], 'chosen_epoch': ck['epoch'], 'chosen_val_bce': ck['val_bce'],
                        'train_seconds': ck['train_seconds'], 'smoke': ck.get('smoke'), 'model_sha256': sha256(cpath)}
    target, mask, counts, test_fp = margin_binding(args.final_points, test, ab.manifests(args.data, 'test')[1])
    result[ARM]['margin_head_test'] = {**margin_report(mp, target, mask), 'counts': counts, 'final_points': test_fp}
    by_game = {}
    for i, gid in enumerate(test['game']):
        by_game.setdefault(int(gid), []).append(i)
    ids = sorted(by_game)
    by_game = {k: np.asarray(v) for k, v in by_game.items()}
    rng = random.Random(g['seed'])
    diffs = []
    for _ in range(g['resamples']):
        sample = np.concatenate([by_game[rng.choice(ids)] for _ in ids])
        diffs.append(ab.fast_auc(p[sample], y[sample]) - ab.fast_auc(pr[sample], y[sample]))
    diffs.sort()
    diff = result[ARM]['auc'] - result['tiles_reference']['auc']
    lcb, ucb = diffs[int(.025 * len(diffs))], diffs[int(.975 * len(diffs)) - 1]
    auc_pass = bool(diff >= g['min_diff'] and lcb > g['lcb_gt'])
    xeval, xeval_pass, xeval_failures, missing = None, None, None, []
    if args.xeval:
        xeval = json.loads(Path(args.xeval).read_text())
        win_pt = args.out / f'{ARM}-win.pt'
        if not win_pt.exists() or xeval['model_sha256'] != sha256(win_pt):
            raise ValueError('xeval was not run on this checkpoint\'s tiles_margin-win.pt')
        derived = torch.load(win_pt, weights_only=False, map_location='cpu')['derived_from']
        if derived['pt_sha256'] != sha256(path):
            raise ValueError('tiles_margin-win.pt was derived from a different tiles_margin.pt')
        xeval_failures = xeval_check(xeval, g['xeval_min'])
        xeval_pass = not xeval_failures
    else:
        missing.append('xeval')
    passed = None if missing else bool(auc_pass and xeval_pass)
    return {'arms': result, 'vs_reference': {'auc_diff': diff, 'ci95': [lcb, ucb], 'pass': auc_pass},
            'reference_auc_declared': g['reference_auc'],
            'reference_auc_matches_declared': abs(result['tiles_reference']['auc'] - g['reference_auc']) < 5e-5,
            'xeval': xeval, 'xeval_pass': xeval_pass, 'xeval_failures': xeval_failures, 'missing': missing, 'pass': passed, 'bar': g,
            'test_rows': len(y), 'test_games': len(ids), 'test_identity': test['identity'],
            'test_arrays_sha256': test['arrays_sha256'], 'train_identity': train_binding['identity'],
            'train_arrays_sha256': train_binding['arrays_sha256'],
            'sealed_after_the_fact': {'train': train_binding['sealed_after_the_fact'],
                                      'test': test['sealed_after_the_fact']},
            'reference_arrays_bound_in_checkpoint': 'train_arrays_sha256' in ref,
            'device': str(device), 'max_rows': args.max_rows, 'smoke': smoke}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'win', 'gate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, default=ab.ROOT / 'experiments' / 'arch-bakeoff.json')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--final-points', type=Path, help='replay_final_points.py output root (one dir per root)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='gate: the arch-bakeoff-v1 tiles.pt')
    parser.add_argument('--xeval', type=Path, help='gate: cross_eval_arch.py output for tiles_margin-win.pt')
    parser.add_argument('--device', default='auto', help='auto (mps if available, else cpu), cpu or mps')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only')
    parser.add_argument('--epochs', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    base = json.loads(args.base_config.read_text())
    check_configs(cfg, base)
    if args.command in ('train', 'gate'):
        if args.data is None:
            parser.error(f'{args.command} needs --data')
        ab.check_data(base, args.data)
    if args.command == 'train' and args.final_points is None:
        parser.error('train needs --final-points')
    if args.command == 'gate' and (args.reference is None or args.final_points is None):
        parser.error('gate needs --reference and --final-points')
    if args.command != 'train' and args.epochs:
        parser.error('--epochs applies to train only')
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / {'train': f'{ARM}-train.json', 'win': f'{ARM}-win.json', 'gate': 'gate.json'}[args.command]
    refuse_existing(output, *((args.out / f'{ARM}.pt',) if args.command == 'train' else ()),
                    *((args.out / f'{ARM}-win.pt',) if args.command == 'win' else ()))
    started = time.monotonic()
    result = {'train': train, 'win': win_checkpoint, 'gate': gate}[args.command](args, cfg, base)
    result.update({'command': args.command, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   'base_config_sha256': hashlib.sha256(args.base_config.read_bytes()).hexdigest(),
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'arch_bakeoff_sha256': hashlib.sha256(Path(ab.__file__).read_bytes()).hexdigest(),
                   'seconds': time.monotonic() - started})
    publish(output, lambda tmp: tmp.write_text(json.dumps(result, indent=1)))
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=1))
    return result


if __name__ == '__main__':
    main()
