"""Within-decision cross-eval of arch-bakeoff checkpoints (predeclared 'next' step of
experiments/arch-bakeoff.json): identical metric, decision filter and bootstrap to
``cross_eval_pv2.py`` (picked candidate's playout win rate minus the served
candidate's on the pv2 test decisions; immediate wins first; city-vs-end report),
with the model loaded through ``arch_bakeoff.make_net`` / ``score``.
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
from train_pv2 import decisions, load  # noqa: E402


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
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
    saved = torch.load(args.model, weights_only=False)
    if saved.get('smoke'):
        raise ValueError('smoke checkpoint')
    arm = saved['arm']
    device = torch.device(args.device)
    net = make_net(torch, arm).to(device)
    net.load_state_dict(saved['state'])
    lo, n = dcfg['split']['test']
    games, corpus = load(args.pv2_data, lo, n, manifest['identity_sha256'])
    rows, cm, cp, cn = [], 0, 0, 0
    for g, cands in decisions(games):
        x = np.asarray([c['x'] for c in cands], dtype=np.float16)
        p, _ = score(torch, net, arm, x, np.arange(len(cands)), saved['mean'], saved['sd'], device)
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
    result = {'arm': arm, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'test_decisions': len(rows), 'pick_minus_served': sum(r[1] for r in rows) / len(rows),
              'pick_minus_served_lcb95': draws[int(.025 * len(draws))],
              'pick_minus_served_ucb95': draws[int(.975 * len(draws)) - 1],
              'city_ranked_above_end_by_model': cm / cn if cn else None, 'city_above_end_in_playouts': cp / cn if cn else None,
              'city_vs_end_decisions': cn, 'test_corpus_sha256': corpus,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
