"""Fit the value-leaf v2 unit conversion on the row 4 TRAIN corpus only.

v1 put the learned win log-odds straight into the build search, where a fixed
purchase bonus in fitted-value units (~8) outweighed the whole log-odds range
(runs/value-leaf-dev-v1.json FAIL). v2 maps the log-odds onto the served leaf's
scale by matching mean and standard deviation over training positions:
``leaf = leaf_mean + (leaf_sd / logit_sd) * (logit - logit_mean)``. Rows whose
served leaf is an immediate win (>= 1e4) are excluded; every other train row
counts once. Test and DEV games are not read.

    .venv/bin/python scripts/fit_value_leaf_scale.py --train runs/value-gate-v1/train \
        --model checkpoints/value_leaf_v1.json --output checkpoints/value_leaf_v2.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.model.read_bytes()
    base = json.loads(raw)
    m = base['model']

    def logit(x):
        return m['bias'] + sum(w * (v - mu) / s for w, v, mu, s in zip(m['weights'], x, m['means'], m['scales']))
    logits, leaves, games = [], [], 0
    for path in sorted(args.train.glob('*.json')):
        game = json.loads(path.read_text())
        if game['truncated']:
            continue
        games += 1
        for row in game['rows']:
            if row['leaf'] < 1e4:
                logits.append(logit(row['features']))
                leaves.append(row['leaf'])
    scale = {'logit_mean': statistics.fmean(logits), 'logit_sd': statistics.pstdev(logits),
             'leaf_mean': statistics.fmean(leaves), 'leaf_sd': statistics.pstdev(leaves),
             'train_games': games, 'train_rows': len(logits)}
    scale['slope'] = scale['leaf_sd'] / scale['logit_sd']
    out = dict(base, schema='value-leaf-v2', base_checkpoint=str(args.model),
               base_checkpoint_sha256=hashlib.sha256(raw).hexdigest(), scale=scale)
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=1)
    print(json.dumps(scale, indent=1))


if __name__ == '__main__':
    main()
