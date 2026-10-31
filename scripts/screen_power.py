"""Report how much paired win-rate signal completed screens actually had.

For every evaluator receipt given, print each panel's seed-cluster count,
cluster SD of the paired win difference, the recorded lift, and the zero-null
sensitivity: the smallest true lift a one-sided test against zero detects at 80%
power with the evaluator's Bonferroni per-panel alpha. That is not the power to
clear a +5 pp bar. The recommendation instead uses the evaluator's admission
check: seeds per panel so every panel clears ``--bar`` with family probability
.8 when the true lift is ``--alternative`` (largest observed SD, floor .25).

Usage:
    uv run python scripts/screen_power.py runs/finish-dev-v1.json runs/opening-search-dev-v1.json
    uv run python scripts/screen_power.py --bar 0.05 --alternative 0.10 --glob 'runs/*-dev-v1.json'
"""

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from catan_ai.power import check_design, cluster_sigma, minimum_detectable_effect, panel_differences  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipts', nargs='*', type=Path)
    parser.add_argument('--glob', help='additional receipts by shell pattern')
    parser.add_argument('--bar', type=float, default=.05, help='point-estimate pass bar of the next screen')
    parser.add_argument('--alternative', type=float, default=.10, help='true lift the next screen should find')
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
                         'zero_null_mde_80': round(minimum_detectable_effect(sigma, len(diffs), alpha), 4), 'pass': summary.get('pass')})
    if not rows:
        print('no readable receipts', file=sys.stderr)
        return 1
    width = max(len(r['receipt']) for r in rows)
    print(f"{'receipt':<{width}}  panel     clusters  sd      lift     bar    zero-null mde80   pass")
    for r in rows:
        lift = 'none' if r['lift'] is None else f"{r['lift']:+.3f}"
        print(f"{r['receipt']:<{width}}  {r['panel']:<9} {r['clusters']:>8}  {r['cluster_sd']:.3f}  {lift:>7}  {r['bar']!s:>5}  {r['zero_null_mde_80']:>15.3f}  {r['pass']}")
    worst = max(sigmas)
    planning = max(.25, worst)
    for panels in (1, 2, 3):
        try:
            record = check_design({'panels': [{}] * panels, 'seeds_per_panel': 1_000_000,
                                   'gates': {'minimum_paired_win_lift': args.bar},
                                   'power': {'alternative_effect': args.alternative, 'planning_sigma': planning}})
        except ValueError as exc:
            print(f'{panels} panel(s): {exc}')
            continue
        print(f"bar {args.bar:+.3f}, true lift {args.alternative:+.3f}, {panels} panel(s): "
              f"{record['required_clusters']} clusters per panel "
              f"(planning sigma {planning:.3f}; largest observed {worst:.3f})")
    if args.json:
        args.json.write_text(json.dumps({'bar': args.bar, 'alternative_effect': args.alternative, 'largest_observed_sd': worst,
                                         'planning_sigma': planning, 'panels': rows}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
