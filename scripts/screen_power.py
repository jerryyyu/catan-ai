"""Report how much paired win-rate signal completed screens actually had.

For every evaluator receipt given, print each panel's seed-cluster count,
cluster SD of the paired win difference, the recorded lift, and the smallest
true lift that design could detect at 80% power with the evaluator's
Bonferroni per-panel alpha. Then recommend seeds per panel for a target effect
using the largest observed SD (and the .25 planning floor used elsewhere).

Usage:
    uv run python scripts/screen_power.py runs/finish-dev-v1.json runs/opening-search-dev-v1.json
    uv run python scripts/screen_power.py --target 0.05 --glob 'runs/*-dev-v1.json'
"""

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from catan_ai.power import cluster_sigma, minimum_detectable_effect, panel_differences, required_clusters  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipts', nargs='*', type=Path)
    parser.add_argument('--glob', help='additional receipts by shell pattern')
    parser.add_argument('--target', type=float, default=.05, help='effect the next screen must detect')
    parser.add_argument('--max-bytes', type=int, default=256_000_000, help='skip larger receipts')
    parser.add_argument('--json', type=Path, help='also write the table as JSON')
    args = parser.parse_args()
    paths = list(args.receipts) + [Path(p) for p in sorted(glob.glob(args.glob))] if args.glob else list(args.receipts)
    rows, sigmas = [], []
    for path in paths:
        if not path.exists() or path.stat().st_size > args.max_bytes:
            print(f'skip {path}: missing or larger than {args.max_bytes} bytes', file=sys.stderr)
            continue
        receipt = json.loads(path.read_text())
        if 'summary' not in receipt or 'games' not in receipt:
            continue
        panels = receipt['config'].get('panels', [])
        alpha = .05 / max(1, len(panels))
        for panel in panels:
            name = panel['name']
            diffs = panel_differences(receipt['games'], name)
            if len(diffs) < 2:
                continue
            sigma = cluster_sigma(diffs)
            sigmas.append(sigma)
            summary = receipt['summary']['panels'].get(name, {})
            rows.append({'receipt': str(path), 'panel': name, 'clusters': len(diffs), 'cluster_sd': round(sigma, 4),
                         'lift': summary.get('paired_win_lift'), 'bar': receipt['config']['gates'].get('minimum_paired_win_lift'),
                         'mde_80': round(minimum_detectable_effect(sigma, len(diffs), alpha), 4), 'pass': summary.get('pass')})
    if not rows:
        print('no readable receipts', file=sys.stderr)
        return 1
    width = max(len(r['receipt']) for r in rows)
    print(f"{'receipt':<{width}}  panel     clusters  sd      lift     bar    mde80   pass")
    for r in rows:
        lift = 'none' if r['lift'] is None else f"{r['lift']:+.3f}"
        print(f"{r['receipt']:<{width}}  {r['panel']:<9} {r['clusters']:>8}  {r['cluster_sd']:.3f}  {lift:>7}  {r['bar']!s:>5}  {r['mde_80']:.3f}  {r['pass']}")
    worst = max(sigmas)
    planning = max(.25, worst)
    for panels in (1, 2, 3):
        alpha = .05 / panels
        print(f"target {args.target:+.3f} with {panels} panel(s): {required_clusters(planning, args.target, alpha)} clusters per panel "
              f"(planning sigma {planning:.3f}; largest observed {worst:.3f})")
    if args.json:
        args.json.write_text(json.dumps({'target_effect': args.target, 'largest_observed_sd': worst,
                                         'planning_sigma': planning, 'panels': rows}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
