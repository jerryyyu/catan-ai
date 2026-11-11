"""Offline gate for placement-value P1 (experiments/placement-value.json ``gate``).

1. placement (primary), on the sealed placement-test-served root: per setup decision the model's top-scored
   candidate's playout win rate minus the served economic choice's; mean over the 400 decisions; game-clustered
   bootstrap (1,000 resamples, seed 37) 5th-percentile lower bound > 0;
2. non-inferiority vs T9 on the original and on-policy end-of-turn test rows: AUC diff lower bound > -0.005.

Reported: the placement metric for T9 and the served net, the same metric on placement-data-v1's test games, the
oracle rate, top-1 agreement with the economic choice, and setupfork test rows vs T9. Anything missing leaves
``pass`` null; a smoke run never passes.

    placement_value_gate.py --runs runs --placetest runs/placement-test-served-v1 \\
        --legacy-test runs/placement-data-v1 --out runs/placement-value-v1/p1-gate.json
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
from artifact_guard import producer_provenance  # noqa: E402
from train_value_v2 import DATA_KEYS, shards  # noqa: E402
import value_arms as va  # noqa: E402
from value_arms_gate import paired_auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / 'experiments' / 'placement-value.json'
PLACETEST_DECL = ROOT / 'experiments' / 'placement-test-served.json'
RESAMPLES, BOOT_SEED, PCT, NONINF = 1000, 37, .05, -0.005
LEGACY_TEST = (2896450, 50)


def load_placetest(root, declaration=PLACETEST_DECL):
    """{(seed, setup_index): [rows]} from a sealed placetest root made under ``declaration``."""
    root = Path(root)
    done = json.loads((root / 'DONE.json').read_text())
    decl = json.loads(Path(declaration).read_text())
    keys = DATA_KEYS + ('policy', 'top_k', 'random_k', 'playouts')
    if any(done['config'].get(k) != decl.get(k) for k in keys):
        raise ValueError(f'{root} was not made under {Path(declaration).name}')
    files = sorted((root / 'rows').glob('*.json'))
    if {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files} != done['shards']:
        raise ValueError(f'{root}: row files differ from the sealed DONE.json')
    out = {}
    for f in files:
        r = json.loads(f.read_text())
        out.setdefault((r['seed'], r['setup_index']), []).append(r)
    lo, n = decl['split']['test']
    if sorted({s for s, _ in out}) != list(range(lo, lo + n)) or any(len(v) != decl['top_k'] + decl['random_k']
                                                                   or sum(r['served'] for r in v) != 1 for v in out.values()):
        raise ValueError(f'{root}: decisions or candidates differ from the declaration')
    return out, hashlib.sha256(json.dumps(done, sort_keys=True).encode()).hexdigest()


def load_legacy(root, test=LEGACY_TEST):
    """placement-data-v1 test games in the same {(seed, index): [rows]} shape (reported only)."""
    out = {}
    for seed in range(test[0], test[0] + test[1]):
        rec = json.loads((Path(root) / 'games' / f'{seed}.json').read_text())
        for d in rec['decisions']:
            out[(seed, d['setup_index'])] = [{'x': c['x'], 'wins': c['wins'], 'playouts': c['playouts'],
                                              'served': c['served']} for c in d['candidates']]
    return out


def placement_metric(decisions, scores):
    """decisions {(seed, i): [rows]}, scores {(seed, i): [score per row]} -> summary with a game-clustered bound."""
    per_game, top1, oracle = {}, [], []
    for key, rows in decisions.items():
        rate = [r['wins'] / r['playouts'] for r in rows]
        served = next(i for i, r in enumerate(rows) if r['served'])
        pick = int(np.argmax(scores[key]))
        per_game.setdefault(key[0], []).append(rate[pick] - rate[served])
        top1.append(pick == served)
        oracle.append(max(rate) - rate[served])
    games = sorted(per_game)
    rng = random.Random(BOOT_SEED)
    flat = {g: per_game[g] for g in games}
    mean = float(np.mean([v for g in games for v in flat[g]]))
    boots = []
    for _ in range(RESAMPLES):
        draw = [v for _ in games for v in flat[rng.choice(games)]]
        boots.append(float(np.mean(draw)))
    boots.sort()
    return {'pick_minus_served': mean, 'lcb': boots[int(PCT * RESAMPLES)], 'decisions': len(top1), 'games': len(games),
            'top1_agrees_with_served': float(np.mean(top1)), 'oracle_minus_served': float(np.mean(oracle))}


def score_decisions(torch, net, saved, decisions, device):
    keys = sorted(decisions)
    x = np.asarray([r['x'] for k in keys for r in decisions[k]], dtype=np.float32)
    p = ab.score(torch, net, 'tiles', x, np.arange(len(x)), saved['mean'].to(device), saved['sd'].to(device), device)[0]
    out, i = {}, 0
    for k in keys:
        out[k] = list(p[i:i + len(decisions[k])])
        i += len(decisions[k])
    return out


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DECL)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--placetest', type=Path, required=True)
    parser.add_argument('--legacy-test', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    smoke = bool(args.max_rows)
    model = args.model or args.runs / 'placement-value-v1' / 'p1.pt'
    new = torch.load(model, weights_only=False, map_location='cpu')
    if new.get('value_arm') != 'p1' or new['config'] != cfg or (new.get('smoke') and not smoke):
        raise ValueError(f'{model} is not a full p1 checkpoint trained under this declaration')
    va.check_training(cfg, 'p1', new, args.runs)
    device = ab.device_of(torch, args.device)
    nets = {}
    for name, ref in (('new', None), ('t9', cfg['reference']), ('served', cfg['served_reference'])):
        if ref is None:
            saved = new
        else:
            path = args.runs / ref['pt']
            if va.sha256(path) != ref['pt_sha256']:
                raise ValueError(f'{name}: {path} is not the declared checkpoint')
            saved = torch.load(path, weights_only=False, map_location='cpu')
        net = ab.make_net(torch, 'tiles', ab.layout(), size=saved.get('size'), phase='phase_vec' in saved['state'])
        net.load_state_dict(saved['state'])
        nets[name] = (net.to(device), saved)
    decisions, placetest_sha = load_placetest(args.placetest)
    placement = {n: placement_metric(decisions, score_decisions(torch, net, s, decisions, device))
                 for n, (net, s) in nets.items()}
    legacy = None
    if args.legacy_test is not None:
        old = load_legacy(args.legacy_test)
        legacy = {n: placement_metric(old, score_decisions(torch, net, s, old, device)) for n, (net, s) in nets.items()}
    parts = {}
    roots = va.check_roots(cfg, args.runs, ['original', 'onpolicy', 'setupfork'])
    for label, (path, split) in roots.items():
        test = ab.subsample(shards(path, 'test'), args.max_rows, 0)
        lo, n = split['test']
        if test['game'].min() < lo or test['game'].max() >= lo + n:
            raise ValueError(f'{label}: test rows outside the declared test split')
        rows = np.arange(len(test['y']))
        preds = {k: ab.score(torch, nets[k][0], 'tiles', test['x'], rows, nets[k][1]['mean'].to(device),
                             nets[k][1]['sd'].to(device), device)[0] for k in ('new', 't9')}
        parts[label] = paired_auc(preds['new'], preds['t9'], test['y'].astype(np.float64), test['game'], pct=PCT,
                                  seed=BOOT_SEED)
    checks = {'placement': bool(placement['new']['lcb'] > 0),
              'original_noninferior': bool(parts['original']['lcb'] > NONINF),
              'onpolicy_noninferior': bool(parts['onpolicy']['lcb'] > NONINF)}
    passed = None if smoke else all(checks.values())
    out = {'arm': 'p1', 'declaration_sha256': va.sha256(args.config), 'script_sha256': va.sha256(__file__),
           'model_sha256': va.sha256(model), 'placetest_done_sha256': placetest_sha, 'placement': placement,
           'legacy_placement': legacy, 'parts_vs_t9': parts, 'checks': checks, 'pass': passed, 'smoke': smoke,
           'bar': {**cfg['gate'], 'resamples': RESAMPLES, 'seed': BOOT_SEED, 'lower_percentile': PCT, 'non_inferiority': NONINF},
           'train_roots': new['train'], 'model_provenance': new.get('provenance'), 'provenance': producer_provenance(),
           'device': str(device)}
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'checks': checks, 'pass': passed, 'placement': {k: v['pick_minus_served'] for k, v in placement.items()}}, indent=1))


if __name__ == '__main__':
    main()
