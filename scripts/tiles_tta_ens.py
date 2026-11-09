"""Transformer T1/T2 (experiments/tiles-tta-ens.json): test-time symmetry averaging and a seed ensemble.

``train-seed --seed K`` trains one more arch-bakeoff ``tiles`` model with experiments/arch-bakeoff.json
unchanged except ``common.seed = K`` (``arch_bakeoff.train``; the checkpoint records that config).

``gate`` scores the declared test rows with the reference (seed 0, the served weights) and each arm:

* ``tta``: reference weights, P(I win) averaged over the 12 hex symmetries of the encoder-v2 input
  (``train_value_v2.symmetry_index``, the permutations training augments with);
* ``ens5``: mean P(I win) of the reference and the four extra seeds;
* ``ens5_tta``: mean over 5 seeds x 12 symmetries.

Each arm is judged alone: AUC(arm) - AUC(reference) >= 0 with a game-clustered bootstrap 95% LCB >
-0.002 (1,000 resamples, seed 29, both from the declaration), and the cross-eval receipts
(``scripts/cross_eval_arch.py`` with the arm's models and ``--tta``) must show the arm's pick - served
>= the reference's on the same pv2 test root, the arm's receipt eligible. Anything missing leaves
``pass`` null.

    tiles_tta_ens.py train-seed --seed 1 --data runs/value-v2-100k-shards --out runs/tiles-ens-v1/seed1
    tiles_tta_ens.py gate --data runs/value-v2-100k-shards --reference runs/arch-bakeoff-v1/tiles.pt \\
        --seeds runs/tiles-ens-v1/seed{1,2,3,4}/tiles.pt --xeval-dir runs/tiles-ens-v1 --out runs/tiles-ens-v1/gate.json
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_bakeoff as ab  # noqa: E402
from train_value_v2 import shards, symmetry_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / 'experiments' / 'tiles-tta-ens.json'
BASE = ROOT / 'experiments' / 'arch-bakeoff.json'
ARMS = ('tta', 'ens5', 'ens5_tta')
RESAMPLES, BOOT_SEED, LCB_FLOOR = 1000, 29, -0.002


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_config(seed: int) -> dict:
    cfg = json.loads(BASE.read_text())
    cfg = copy.deepcopy(cfg)
    cfg['common']['seed'] = seed
    return cfg


def train_seed(args):
    if args.seed < 1:
        raise ValueError('seed 0 is the reference; train seeds 1-4')
    args.out.mkdir(parents=True, exist_ok=False)
    ns = SimpleNamespace(arm='tiles', data=args.data, records=None, out=args.out, device=args.device,
                         max_rows=args.max_rows, epochs=args.epochs)
    result = ab.train(ns, seed_config(args.seed))
    (args.out / 'tiles-train.json').write_text(json.dumps({**result, 'seed': args.seed,
                                                            'declaration_sha256': sha256(DECL)}, indent=1))
    print(json.dumps({k: result[k] for k in ('chosen_epoch', 'chosen_val_bce', 'train_seconds', 'model_sha256')}))


def load_net(torch, path, device, expect_seed=None, allow_smoke=False):
    saved = torch.load(path, weights_only=False, map_location='cpu')
    if saved['arm'] != 'tiles' or (saved.get('smoke') and not allow_smoke):
        raise ValueError(f'{path}: not a full tiles checkpoint')
    if expect_seed is not None and saved['config'] != seed_config(expect_seed):
        raise ValueError(f'{path}: config is not arch-bakeoff.json with common.seed={expect_seed}')
    net = ab.make_net(torch, 'tiles', ab.layout())
    net.load_state_dict(saved['state'])
    net.to(device)
    return net, saved


def predict(torch, nets, x, rows, device, tta, perms):
    """Mean P(I win) over the given (net, saved) models and, with ``tta``, the 12 symmetries."""
    total, count = np.zeros(len(rows)), 0
    for net, saved in nets:
        mean, sd = saved['mean'].to(device), saved['sd'].to(device)
        for perm in (perms if tta else [None]):
            xp = x if perm is None else x[:, perm]
            p, _ = ab.score(torch, net, 'tiles', xp, rows, mean, sd, device)
            total += p
            count += 1
    return total / count


def gate(args):
    import torch
    device = ab.device_of(torch, args.device)
    decl = json.loads(DECL.read_text())
    cfg = json.loads(BASE.read_text())
    test = ab.subsample(shards(args.data, 'test'), args.max_rows, cfg['common']['seed'])
    lo, n = ab.declared_split(cfg)['test']
    if test['game'].min() < lo or test['game'].max() >= lo + n:
        raise ValueError('test shards hold games outside the declared test split')
    rows = np.arange(len(test['y']))
    y = test['y'].astype(np.float64)
    perms = symmetry_index(torch).numpy()
    if len(perms) != 12:
        raise ValueError(f'expected 12 symmetries, got {len(perms)}')
    smoke = bool(args.max_rows)
    ref = load_net(torch, args.reference, device, allow_smoke=smoke)
    seeds = [load_net(torch, p, device, expect_seed=k, allow_smoke=smoke) for k, p in enumerate(args.seeds, start=1)]
    if len(seeds) != 4:
        raise ValueError('the ensemble declares four extra seeds')
    preds = {'reference': predict(torch, [ref], test['x'], rows, device, False, perms),
             'tta': predict(torch, [ref], test['x'], rows, device, True, perms),
             'ens5': predict(torch, [ref] + seeds, test['x'], rows, device, False, perms),
             'ens5_tta': predict(torch, [ref] + seeds, test['x'], rows, device, True, perms)}
    per_seed = {f'seed{k}': ab.fast_auc(predict(torch, [s], test['x'], rows, device, False, perms), y)
                for k, s in enumerate(seeds, start=1)}
    result = {name: {'auc': ab.fast_auc(p, y), 'log_loss': ab.log_loss(p, y), 'ece10': ab.ece(p, y)}
              for name, p in preds.items()}
    by_game = {}
    for i, g in enumerate(test['game']):
        by_game.setdefault(int(g), []).append(i)
    ids = sorted(by_game)
    by_game = {g: np.asarray(v) for g, v in by_game.items()}
    rng = random.Random(BOOT_SEED)
    diffs = {a: [] for a in ARMS}
    for _ in range(RESAMPLES):
        sample = np.concatenate([by_game[rng.choice(ids)] for _ in ids])
        b = ab.fast_auc(preds['reference'][sample], y[sample])
        for a in ARMS:
            diffs[a].append(ab.fast_auc(preds[a][sample], y[sample]) - b)
    ref_x = json.loads((args.xeval_dir / 'xeval-reference.json').read_text()) if (args.xeval_dir / 'xeval-reference.json').exists() else None
    for a in ARMS:
        d = sorted(diffs[a])
        diff = result[a]['auc'] - result['reference']['auc']
        lcb, ucb = d[int(.025 * len(d))], d[int(.975 * len(d)) - 1]
        auc_pass = bool(diff >= 0 and lcb > LCB_FLOOR)
        xpath = args.xeval_dir / f'xeval-{a}.json'
        xe = json.loads(xpath.read_text()) if xpath.exists() else None
        x_pass, x_fail = None, []
        if xe is not None and ref_x is not None:
            if xe['test_corpus_sha256'] != ref_x['test_corpus_sha256']:
                x_fail.append('cross-evals on different pv2 test corpora')
            if not xe['eligible_for_next_gate']:
                x_fail.append(f'arm cross-eval ineligible: {xe["ineligible_reason"]}')
            if xe['pick_minus_served'] < ref_x['pick_minus_served']:
                x_fail.append(f'pick_minus_served {xe["pick_minus_served"]} < reference {ref_x["pick_minus_served"]}')
            x_pass = not x_fail
        result[a].update(auc_diff=diff, auc_diff_ci95=[lcb, ucb], auc_pass=auc_pass,
                         xeval=xe and {k: xe[k] for k in ('pick_minus_served', 'pick_minus_served_lcb95',
                                                          'pick_minus_served_ucb95', 'eligible_for_next_gate',
                                                          'city_ranked_above_end_by_model', 'model_sha256')},
                         xeval_pass=x_pass, xeval_failures=x_fail,
                         **{'pass': (auc_pass and x_pass) if x_pass is not None else None})
    passing = [a for a in ARMS if result[a]['pass']]
    pick = max(passing, key=lambda a: (result[a]['xeval']['pick_minus_served'], -ARMS.index(a))) if passing else None
    out = {'declaration_sha256': sha256(DECL), 'base_config_sha256': sha256(BASE), 'script_sha256': sha256(__file__),
           'reference': str(args.reference), 'reference_sha256': sha256(args.reference),
           'seeds': [str(p) for p in args.seeds], 'seed_sha256': [sha256(p) for p in args.seeds],
           'per_seed_auc': per_seed, 'arms': result, 'reference_xeval': ref_x and {k: ref_x[k] for k in ('pick_minus_served', 'pick_minus_served_lcb95', 'model_sha256')},
           'test_rows': len(y), 'test_games': len(ids), 'test_identity': test['identity'], 'test_arrays_sha256': test['arrays_sha256'],
           'sealed_after_the_fact': test['sealed_after_the_fact'], 'bar': {'auc_diff_min': 0, 'lcb_gt': LCB_FLOOR, 'resamples': RESAMPLES, 'seed': BOOT_SEED},
           'screen_candidate': pick, 'smoke': bool(args.max_rows), 'device': str(device),
           'declared_screen': decl['screen']}
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({a: {k: result[a].get(k) for k in ('auc', 'auc_diff', 'auc_diff_ci95', 'pass')} for a in ('reference',) + ARMS}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train-seed', 'gate'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--seeds', type=Path, nargs='*', default=[])
    parser.add_argument('--xeval-dir', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only')
    parser.add_argument('--epochs', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    if args.command == 'train-seed':
        train_seed(args)
    else:
        gate(args)


if __name__ == '__main__':
    main()
