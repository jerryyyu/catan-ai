"""Offline gate for the on-policy retrain (experiments/tiles-onpolicy.json, board #1 T3).

The candidate is the arch-bakeoff ``tiles`` arm trained by ``scripts/arch_bakeoff.py train --config
experiments/tiles-onpolicy.json --arm tiles`` on the re-encoded on-policy shards; the reference is the served
checkpoint (``pt_sha256`` in checkpoints/value_tiles_v1.meta.json). All three parts must hold:

1. on-policy test rows (the new shards' test split): AUC(new) - AUC(ref) >= 0, game-clustered bootstrap 95%
   LCB > -0.002;
2. original test rows (runs/value-v2-100k-shards test split): non-inferior, bootstrap LCB > -0.005;
3. cross-eval on the declared pv2 test root: new pick - served >= ref pick - served - 0.002, new receipt
   eligible. Both receipts are bound (model hash, no TTA, this tree's cross_eval_arch.py, the corpus
   recomputed from ``--pv2-data``).

Bootstrap: 1,000 resamples, seed 29. Anything missing leaves ``pass`` null; a smoke run never passes.

    tiles_onpolicy_gate.py --new runs/tiles-onpolicy-v1/tiles.pt --reference runs/arch-bakeoff-v1/tiles.pt \\
        --onpolicy-data runs/value-v2-tiles-shards --original-data runs/value-v2-100k-shards \\
        --xeval runs/tiles-onpolicy-v1/xeval.json --reference-xeval runs/tiles-onpolicy-v1/xeval-reference.json \\
        --pv2-data runs/pv2-test-v2 --out runs/tiles-onpolicy-v1/gate.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_bakeoff as ab  # noqa: E402
from train_value_v2 import shards  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / 'experiments' / 'tiles-onpolicy.json'
SERVED_META = ROOT / 'checkpoints' / 'value_tiles_v1.meta.json'
XEVAL_SCRIPT = Path(__file__).resolve().parent / 'cross_eval_arch.py'
PV2_DATA_CONFIG = ROOT / 'experiments' / 'pv2-test-v2-data.json'
RESAMPLES, BOOT_SEED = 1000, 29
ONPOLICY_LCB, ORIGINAL_LCB, XEVAL_MARGIN = -0.002, -0.005, 0.002


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_auc(p_new, p_ref, y, games):
    by_game = {}
    for i, g in enumerate(games):
        by_game.setdefault(int(g), []).append(i)
    ids = sorted(by_game)
    idx = {g: np.asarray(v) for g, v in by_game.items()}
    rng = random.Random(BOOT_SEED)
    diffs = []
    for _ in range(RESAMPLES):
        s = np.concatenate([idx[rng.choice(ids)] for _ in ids])
        diffs.append(ab.fast_auc(p_new[s], y[s]) - ab.fast_auc(p_ref[s], y[s]))
    diffs.sort()
    a_new, a_ref = ab.fast_auc(p_new, y), ab.fast_auc(p_ref, y)
    return {'auc_new': a_new, 'auc_ref': a_ref, 'diff': a_new - a_ref,
            'ci95': [diffs[int(.025 * len(diffs))], diffs[int(.975 * len(diffs)) - 1]],
            'log_loss_new': ab.log_loss(p_new, y), 'log_loss_ref': ab.log_loss(p_ref, y),
            'ece10_new': ab.ece(p_new, y), 'ece10_ref': ab.ece(p_ref, y), 'rows': len(y), 'games': len(ids)}


def bind_xeval(receipt, model_sha, script_sha, corpus):
    problems = []
    if receipt is None:
        return ['receipt missing']
    if receipt.get('models_sha256', [receipt.get('model_sha256')]) != [model_sha]:
        problems.append(f'models {receipt.get("models_sha256")} != [{model_sha}]')
    if receipt.get('tta'):
        problems.append('cross-eval used symmetry averaging')
    if receipt.get('script_sha256') != script_sha:
        problems.append('cross-eval script differs from this tree')
    if corpus is None or receipt.get('test_corpus_sha256') != corpus:
        problems.append('not on the declared pv2 test corpus (or corpus not recomputed)')
    return problems


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('new', 'reference', 'onpolicy-data', 'original-data', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('xeval', 'reference-xeval', 'pv2-data'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    decl = json.loads(DECL.read_text())
    device = ab.device_of(torch, args.device)
    smoke = bool(args.max_rows)
    new = torch.load(args.new, weights_only=False, map_location='cpu')
    if new['arm'] != 'tiles' or new['config'] != decl or (new.get('smoke') and not smoke):
        raise ValueError('candidate is not a full tiles checkpoint trained under experiments/tiles-onpolicy.json')
    ref_sha, served_pt = sha256(args.reference), json.loads(SERVED_META.read_text())['pt_sha256']
    if not smoke and ref_sha != served_pt:
        raise ValueError(f'reference {ref_sha} is not the served checkpoint {served_pt}')
    ref = torch.load(args.reference, weights_only=False, map_location='cpu')
    nets = {}
    for name, saved in (('new', new), ('ref', ref)):
        net = ab.make_net(torch, 'tiles', ab.layout())
        net.load_state_dict(saved['state'])
        nets[name] = (net.to(device), saved)
    parts = {}
    for label, root in (('onpolicy', args.onpolicy_data), ('original', args.original_data)):
        test = ab.subsample(shards(root, 'test'), args.max_rows, 0)
        rows = np.arange(len(test['y']))
        preds = {n: ab.score(torch, net, 'tiles', test['x'], rows, s['mean'].to(device), s['sd'].to(device), device)[0]
                 for n, (net, s) in nets.items()}
        parts[label] = {**paired_auc(preds['new'], preds['ref'], test['y'].astype(np.float64), test['game']),
                        'test_identity': test['identity'], 'test_arrays_sha256': test['arrays_sha256']}
    corpus = None
    if args.pv2_data is not None:
        from tiles_margin import pv2_corpus
        corpus = pv2_corpus(args.pv2_data, PV2_DATA_CONFIG)
    xe = json.loads(args.xeval.read_text()) if args.xeval and args.xeval.exists() else None
    xr = json.loads(args.reference_xeval.read_text()) if args.reference_xeval and args.reference_xeval.exists() else None
    script = sha256(XEVAL_SCRIPT)
    binding = [f'new: {p}' for p in bind_xeval(xe, sha256(args.new), script, corpus)] + \
              [f'reference: {p}' for p in bind_xeval(xr, ref_sha, script, corpus)]
    on, orig = parts['onpolicy'], parts['original']
    checks = {'onpolicy_auc': bool(on['diff'] >= 0 and on['ci95'][0] > ONPOLICY_LCB),
              'original_auc': bool(orig['ci95'][0] > ORIGINAL_LCB)}
    if binding:
        checks['xeval'] = None
    else:
        checks['xeval'] = bool(xe['eligible_for_next_gate']
                               and xe['pick_minus_served'] >= xr['pick_minus_served'] - XEVAL_MARGIN)
    passed = None if smoke or None in checks.values() else all(checks.values())
    out = {'declaration_sha256': sha256(DECL), 'script_sha256': sha256(__file__), 'new_sha256': sha256(args.new),
           'reference_sha256': ref_sha, 'served_pt_sha256': served_pt, 'parts': parts, 'checks': checks,
           'xeval': xe and {k: xe.get(k) for k in ('pick_minus_served', 'pick_minus_served_lcb95', 'eligible_for_next_gate',
                                                   'city_ranked_above_end_by_model')},
           'reference_xeval': xr and {k: xr.get(k) for k in ('pick_minus_served', 'pick_minus_served_lcb95')},
           'xeval_binding_problems': binding, 'pv2_corpus_sha256': corpus, 'pass': passed, 'smoke': smoke,
           'bar': {'onpolicy_lcb_gt': ONPOLICY_LCB, 'original_lcb_gt': ORIGINAL_LCB, 'xeval_margin': XEVAL_MARGIN,
                   'resamples': RESAMPLES, 'seed': BOOT_SEED}, 'device': str(device)}
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'checks': checks, 'pass': passed, 'onpolicy_diff': on['diff'], 'original_diff': orig['diff']}, indent=1))


if __name__ == '__main__':
    main()
