"""Within-decision cross-eval of arch-bakeoff checkpoints (predeclared 'next' step of
experiments/arch-bakeoff.json): identical metric, decision filter and bootstrap to
``cross_eval_pv2.py`` (picked candidate's playout win rate minus the served
candidate's on the pv2 test decisions; immediate wins first; city-vs-end report),
with the model loaded through ``arch_bakeoff.make_net`` / ``score``.

Truncation (Codex HOLD @b48f61d, P2): the report carries the rollout-truncation status
of the test games' candidates from the shared ``train_pv2.truncation`` check (as
``cross_eval_pv2`` and ``train_pv2.gate``). A censored rollout in pre-fix pv2 data is
indistinguishable from a loss, so ``eligible_for_next_gate`` is true only when the
lift's 95% LCB is > 0 *and* truncation is recorded and zero; otherwise it is false with
the reason in ``ineligible_reason``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_bakeoff import make_net, score  # noqa: E402
from train_pv2 import decisions, load, truncation  # noqa: E402


def ineligible_reason(lcb, censored):
    """Why the lift does not qualify for the next gate (None: it qualifies). Truncation that is
    unrecorded or present disqualifies whatever the CI says."""
    if censored['violation'] is not None:
        return f"rollout truncation: {censored['violation']}"
    return None if lcb > 0 else 'pick_minus_served_lcb95 <= 0'


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, nargs='+', required=True,
                        help='one checkpoint, or several for a mean-probability ensemble (experiments/tiles-tta-ens.json)')
    parser.add_argument('--tta', action='store_true',
                        help='average P(I win) over the 12 hex symmetries of the encoder-v2 input')
    parser.add_argument('--pv2-data-config', type=Path, required=True)
    parser.add_argument('--pv2-train-config', type=Path, required=True)
    parser.add_argument('--pv2-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    dcfg = json.loads(args.pv2_data_config.read_text())
    tcfg = json.loads(args.pv2_train_config.read_text())
    manifest = json.loads((args.pv2_data / 'manifest.json').read_text())
    if manifest['config'] != dcfg:
        raise ValueError('pv2 data root was produced with a different data config')
    device = torch.device(args.device)
    models = []
    for path in args.model:
        saved = torch.load(path, weights_only=False)
        if saved.get('smoke'):
            raise ValueError('smoke checkpoint')
        net = make_net(torch, saved['arm']).to(device)
        net.load_state_dict(saved['state'])
        models.append((net, saved))
    arm = models[0][1]['arm']
    if any(s['arm'] != arm for _, s in models):
        raise ValueError('an ensemble mixes architectures')
    perms = [None]
    if args.tta:
        from train_value_v2 import symmetry_index
        perms = list(symmetry_index(torch).numpy())
    lo, n = dcfg['split']['test']
    games, corpus = load(args.pv2_data, lo, n, manifest['identity_sha256'])
    rows, cm, cp, cn = [], 0, 0, 0
    for g, cands in decisions(games):
        live = [i for i, c in enumerate(cands) if not c['won_now']]
        p = {}
        if live:
            x = np.asarray([cands[i]['x'] for i in live], dtype=np.float16)
            total = np.zeros(len(live))
            for net, saved in models:
                for perm in perms:
                    out, _ = score(torch, net, arm, x if perm is None else x[:, perm], np.arange(len(live)),
                                   saved['mean'], saved['sd'], device)
                    total += out
            p = dict(zip(live, total / (len(models) * len(perms))))
        scores = [float('inf') if c['won_now'] else float(p[i]) for i, c in enumerate(cands)]
        rate = [c['wins'] / c['playouts'] if c['playouts'] else 1. for c in cands]
        served = next(i for i, c in enumerate(cands) if c['served'])
        picked = max(range(len(cands)), key=lambda i: (scores[i], -i))
        rows.append((g, rate[picked] - rate[served]))
        end = next((i for i, c in enumerate(cands) if not c['seq']), None)
        city = next((i for i, c in enumerate(cands) if any(a['type'] == 'build_city' for a in c['seq'])), None)
        if end is not None and city is not None:
            cn += 1
            cm += scores[city] > scores[end]
            cp += rate[city] > rate[end]
    by_game = {}
    for r in rows:
        by_game.setdefault(r[0], []).append(r)
    ids = sorted(by_game)
    rng = random.Random(tcfg['gate']['bootstrap_seed'])
    draws = sorted(sum(r[1] for r in s) / len(s) for s in
                   ([r for gid in (rng.choice(ids) for _ in ids) for r in by_game[gid]]
                    for _ in range(tcfg['gate']['bootstrap_resamples'])))
    lcb = draws[int(.025 * len(draws))]
    censored = truncation(games)
    reason = ineligible_reason(lcb, censored)
    hashes = [hashlib.sha256(m.read_bytes()).hexdigest() for m in args.model]
    result = {'arm': arm, 'model_sha256': hashes[0] if len(hashes) == 1 else hashlib.sha256(''.join(hashes).encode()).hexdigest(),
              'models_sha256': hashes, 'tta': args.tta,
              'test_decisions': len(rows), 'pick_minus_served': sum(r[1] for r in rows) / len(rows),
              'pick_minus_served_lcb95': lcb,
              'pick_minus_served_ucb95': draws[int(.975 * len(draws)) - 1],
              'city_ranked_above_end_by_model': cm / cn if cn else None, 'city_above_end_in_playouts': cp / cn if cn else None,
              'city_vs_end_decisions': cn, 'test_corpus_sha256': corpus,
              'truncation': censored, 'eligible_for_next_gate': reason is None, 'ineligible_reason': reason,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
