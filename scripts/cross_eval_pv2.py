"""Predeclared cross-eval: score pv2 test decisions with a state-value model.

``experiments/value-v2-100k*.json`` predeclared that their models are also scored
with the pv2 within-decision gate: at each pv2b test decision the model rates each
candidate's afterstate features (encoder v2, chooser's information) and picks the
best (an immediate win first); the metric is the picked candidate's playout win
rate minus the served candidate's, with a game-clustered bootstrap 95% LCB, plus
the city-vs-end ordering report. Same decision filter and bootstrap as
``train_pv2.gate``. Reads only the pv2 test split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pv2 import decisions, load, truncation  # noqa: E402
from train_value_v2 import build_model  # noqa: E402


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--pv2-data-config', type=Path, required=True)
    parser.add_argument('--pv2-train-config', type=Path, required=True)
    parser.add_argument('--pv2-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    vcfg = json.loads(args.value_config.read_text())
    dcfg = json.loads(args.pv2_data_config.read_text())
    tcfg = json.loads(args.pv2_train_config.read_text())
    manifest = json.loads((args.pv2_data / 'manifest.json').read_text())
    if manifest['config'] != dcfg:
        raise ValueError('pv2 data root was produced with a different data config')
    saved = torch.load(args.model, weights_only=False)
    if saved['config'] != vcfg:
        raise ValueError('model was not trained under this value config')
    net = build_model(torch, len(saved['mean']), vcfg['model'])
    net.load_state_dict(saved['state'])
    net.eval()
    lo, n = dcfg['split']['test']
    games, corpus = load(args.pv2_data, lo, n, manifest['identity_sha256'])
    rows, cm, cp, cn = [], 0, 0, 0
    with torch.no_grad():
        for g, cands in decisions(games):
            scores = [float('inf') if c['won_now'] else
                      net(((torch.tensor([c['x']]) - saved['mean']) / saved['sd'])).item() for c in cands]
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
    result = {'value_config': str(args.value_config), 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'test_decisions': len(rows), 'pick_minus_served': sum(r[1] for r in rows) / len(rows),
              'pick_minus_served_lcb95': draws[int(.025 * len(draws))],
              'city_ranked_above_end_by_model': cm / cn if cn else None, 'city_above_end_in_playouts': cp / cn if cn else None,
              'city_vs_end_decisions': cn, 'test_corpus_sha256': corpus,
              'truncation': truncation(games),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
