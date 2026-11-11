"""Offline gate for one value-arms arm (experiments/value-arms.json ``gate``).

Every arm, against the served reference on the same rows (game-clustered bootstrap, 1,000 resamples, seed 31,
one-sided 1st-percentile bound = Bonferroni over 5 arms at .05):

1. original test (value-v2-100k test split): AUC(new) - AUC(ref) lower bound > -0.005;
2. on-policy test (value-v2-tiles test split): lower bound > -0.005;
3. cross-eval on runs/pv2-test-v2: new pick - served >= reference pick - served - 0.002, new receipt eligible,
   both receipts bound to this tree's cross_eval_arch.py, their models and the recomputed corpus;
4. t7/t8/t9: on-policy lower bound > 0;  s1/s2: city_ranked_above_end_by_model <= 0.80.

Reported: AUC, log loss, ECE10 per test set; the end-now diagnostic on endnow-v1 test rows when that root is
complete. Anything missing leaves ``pass`` null; a smoke run never passes.

    value_arms_gate.py --arm t9 --runs runs --xeval runs/value-arms-v1/t9-xeval.json \\
        --reference-xeval runs/value-arms-v1/reference-xeval.json --pv2-data runs/pv2-test-v2 --out runs/value-arms-v1/t9-gate.json
"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_bakeoff as ab  # noqa: E402
from tiles_onpolicy_gate import bind_xeval  # noqa: E402
from train_value_v2 import shards  # noqa: E402
import value_arms as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XEVAL_SCRIPT = Path(__file__).resolve().parent / 'cross_eval_arch.py'
PV2_DATA_CONFIG = ROOT / 'experiments' / 'pv2-test-v2-data.json'
RESAMPLES, BOOT_SEED, PCT = 1000, 31, .01


def paired_auc(p_new, p_ref, y, games, pct=PCT, seed=BOOT_SEED):
    by_game = {}
    for i, g in enumerate(games):
        by_game.setdefault(int(g), []).append(i)
    ids = sorted(by_game)
    idx = {g: np.asarray(v) for g, v in by_game.items()}
    rng = random.Random(seed)
    diffs = []
    for _ in range(RESAMPLES):
        s = np.concatenate([idx[rng.choice(ids)] for _ in ids])
        diffs.append(ab.fast_auc(p_new[s], y[s]) - ab.fast_auc(p_ref[s], y[s]))
    diffs.sort()
    a_new, a_ref = ab.fast_auc(p_new, y), ab.fast_auc(p_ref, y)
    return {'auc_new': a_new, 'auc_ref': a_ref, 'diff': a_new - a_ref, 'lcb_1pct': diffs[int(PCT * RESAMPLES)], 'lcb': diffs[int(pct * RESAMPLES)], 'lcb_pct': pct,
            'log_loss_new': ab.log_loss(p_new, y), 'log_loss_ref': ab.log_loss(p_ref, y),
            'ece10_new': ab.ece(p_new, y), 'ece10_ref': ab.ece(p_ref, y), 'rows': len(y), 'games': len(ids)}


def verdict(arm, gate, parts, xe, xr, binding):
    """{check: True/False/None} and the arm's pass (None when anything is missing)."""
    allb = gate['all_arms']
    checks = {'original_auc': None if 'original' not in parts else bool(parts['original']['lcb_1pct'] > allb['original_test_lcb_gt']),
              'onpolicy_auc': None if 'onpolicy' not in parts else bool(parts['onpolicy']['lcb_1pct'] > allb['onpolicy_test_lcb_gt'])}
    ok = not binding and xe is not None and xr is not None
    checks['xeval'] = bool(xe['eligible_for_next_gate'] and xe['pick_minus_served'] >= xr['pick_minus_served'] - allb['xeval_margin']) if ok else None
    if arm in gate['capacity_arms']['arms']:
        checks['onpolicy_gain'] = None if 'onpolicy' not in parts else bool(parts['onpolicy']['lcb_1pct'] > gate['capacity_arms']['onpolicy_test_diff_lcb_gt'])
    elif arm in gate['support_arms']['arms']:
        rate = xe.get('city_ranked_above_end_by_model') if ok else None
        checks['city_over_end'] = None if rate is None else bool(rate <= gate['support_arms']['city_ranked_above_end_max'])
    else:
        raise ValueError(f'arm {arm} has no gate class')
    return checks, (None if None in checks.values() else all(checks.values()))


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=va.DECL)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='default: runs/value-arms-v1/<arm>.pt')
    for name in ('xeval', 'reference-xeval', 'pv2-data'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    va.arm_spec(cfg, args.arm)
    gate = cfg['gate']
    smoke = bool(args.max_rows)
    model = args.model or args.runs / 'value-arms-v1' / f'{args.arm}.pt'
    new = torch.load(model, weights_only=False, map_location='cpu')
    if new.get('value_arm') != args.arm or new['config'] != cfg or (new.get('smoke') and not smoke):
        raise ValueError(f'{model} is not a full {args.arm} checkpoint trained under this declaration')
    ref_path = args.runs / cfg['reference']['pt']
    if va.sha256(ref_path) != cfg['reference']['pt_sha256']:
        raise ValueError('reference is not the served checkpoint')
    ref = torch.load(ref_path, weights_only=False, map_location='cpu')
    device = ab.device_of(torch, args.device)
    nets = {}
    for name, saved in (('new', new), ('ref', ref)):
        net = ab.make_net(torch, 'tiles', ab.layout(), size=saved.get('size'), phase='phase_vec' in saved['state'])
        net.load_state_dict(saved['state'])
        nets[name] = (net.to(device), saved)
    va.check_training(cfg, args.arm, new, args.runs)  # every training root, exactly (Codex HOLD @e19e5a1)
    roots = va.check_roots(cfg, args.runs, ['original', 'onpolicy'])
    if (args.runs / cfg['roots']['endnow']['shards'] / 'DONE.json').exists():
        roots.update(va.check_roots(cfg, args.runs, ['endnow']))  # end-now diagnostic, reported once complete
    parts = {}
    for label, (path, split) in roots.items():
        test = ab.subsample(shards(path, 'test'), args.max_rows, 0)
        lo, n = split['test']
        if test['game'].min() < lo or test['game'].max() >= lo + n:
            raise ValueError(f'{label}: test rows outside the declared test split')
        rows = np.arange(len(test['y']))
        preds = {k: ab.score(torch, net, 'tiles', test['x'], rows, s['mean'].to(device), s['sd'].to(device), device)[0]
                 for k, (net, s) in nets.items()}
        parts[label] = {**paired_auc(preds['new'], preds['ref'], test['y'].astype(np.float64), test['game']),
                        'test_identity': test['identity'], 'test_arrays_sha256': test['arrays_sha256']}
    diagnostic = parts.pop('endnow', None)
    corpus = None
    if args.pv2_data is not None:
        from tiles_margin import pv2_corpus
        corpus = pv2_corpus(args.pv2_data, PV2_DATA_CONFIG)
    xe = json.loads(args.xeval.read_text()) if args.xeval and args.xeval.exists() else None
    xr = json.loads(args.reference_xeval.read_text()) if args.reference_xeval and args.reference_xeval.exists() else None
    script = va.sha256(XEVAL_SCRIPT)
    binding = [f'new: {p}' for p in bind_xeval(xe, va.sha256(model), script, corpus)] + \
              [f'reference: {p}' for p in bind_xeval(xr, cfg['reference']['pt_sha256'], script, corpus)]
    checks, passed = verdict(args.arm, gate, parts, xe, xr, binding)
    if smoke:
        passed = None
    out = {'arm': args.arm, 'declaration_sha256': va.sha256(args.config), 'script_sha256': va.sha256(__file__),
           'model_sha256': va.sha256(model), 'reference_sha256': cfg['reference']['pt_sha256'], 'parts': parts,
           'endnow_diagnostic': diagnostic, 'checks': checks, 'pass': passed, 'smoke': smoke,
           'xeval': xe and {k: xe.get(k) for k in ('pick_minus_served', 'pick_minus_served_lcb95', 'eligible_for_next_gate',
                                                   'city_ranked_above_end_by_model')},
           'reference_xeval': xr and {k: xr.get(k) for k in ('pick_minus_served', 'city_ranked_above_end_by_model')},
           'xeval_binding_problems': binding, 'pv2_corpus_sha256': corpus, 'params': new.get('params'),
           'bar': {**gate, 'resamples': RESAMPLES, 'seed': BOOT_SEED, 'lower_percentile': PCT}, 'device': str(device),
           'train_roots': new['train'], 'model_provenance': new.get('provenance'), 'provenance': va.provenance()}
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'arm': args.arm, 'checks': checks, 'pass': passed}, indent=1))


if __name__ == '__main__':
    main()
