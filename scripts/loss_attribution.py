"""Where does the candidate lose? Pass 1: what completed receipts already record.

Streams an evaluator receipt (any size; the 1.3 GB held-out files included) and
attributes candidate losses using only the terminal fields each game row
carries: winner, turns, final points per seat and truncation. Replaying games
for event-level causes (robber, discards, awards) is pass 2 and needs compute.

For each panel it reports, for the candidate arm and the control arm:
- win rate by seat (does the candidate lose from particular seats?)
- loss margin distribution: points behind the winner (1, 2, 3+)
- the share of losses where the candidate finished second, i.e. close races
- mean turns in wins and in losses (does the candidate lose long games?)
- who wins when the candidate does not, by opponent seat offset

Paired view: boards where the control arm won from this seat and the candidate
did not, and the reverse. These are the boards worth replaying in pass 2.

Usage:
    uv run python scripts/loss_attribution.py runs/economic-parameters-tactical-heldout-v1.json \
        --output runs/loss-attribution-tactical-heldout-v1.json
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

try:
    import ijson
except ImportError:  # pragma: no cover
    ijson = None


def iter_games(path):
    """Yield game rows without loading the whole receipt into memory."""
    if ijson is None:
        yield from json.loads(path.read_text())['games']
        return
    with path.open('rb') as stream:
        for row in ijson.items(stream, 'games.item', use_float=True):
            yield row


def read_header(path):
    """Config and summary without the games array."""
    if ijson is None:
        data = json.loads(path.read_text())
        return data.get('config', {}), data.get('summary', {})
    config, summary = {}, {}
    with path.open('rb') as stream:
        for prefix, event, _ in ijson.parse(stream):
            if prefix == 'games' and event == 'start_array':
                break
    with path.open('rb') as stream:
        config = next(ijson.items(stream, 'config'), {})
    with path.open('rb') as stream:
        summary = next(ijson.items(stream, 'summary'), {})
    return config, summary


def arm_stats():
    return {'games': 0, 'wins': 0, 'truncated': 0, 'wins_by_seat': Counter(), 'games_by_seat': Counter(),
            'loss_margin': Counter(), 'second_place_losses': 0, 'turns_win': [], 'turns_loss': [],
            'winner_offset': Counter(), 'points_when_lost': []}


def record(stats, row, arm):
    result = row[arm]
    seat = int(row['seat'])
    stats['games'] += 1
    stats['games_by_seat'][seat] += 1
    if result['truncated'] or result['winner'] is None:
        stats['truncated'] += 1
        return
    winner = int(result['winner'])
    points = [int(p) for p in result['points']]
    turns = int(result['turns'])
    if winner == seat:
        stats['wins'] += 1
        stats['wins_by_seat'][seat] += 1
        stats['turns_win'].append(turns)
        return
    stats['turns_loss'].append(turns)
    stats['points_when_lost'].append(points[seat])
    margin = points[winner] - points[seat]
    stats['loss_margin']['1' if margin <= 1 else '2' if margin == 2 else '3+'] += 1
    others = sorted(points[s] for s in range(4) if s != winner)
    if points[seat] == others[-1]:
        stats['second_place_losses'] += 1
    stats['winner_offset'][(winner - seat) % 4] += 1


def finish(stats):
    losses = stats['games'] - stats['wins'] - stats['truncated']
    out = {'games': stats['games'], 'wins': stats['wins'], 'losses': losses, 'truncated': stats['truncated'],
           'win_rate': stats['wins'] / stats['games'] if stats['games'] else None,
           'win_rate_by_seat': {str(s): stats['wins_by_seat'][s] / stats['games_by_seat'][s]
                                for s in sorted(stats['games_by_seat'])},
           'loss_margin_share': {k: v / losses for k, v in sorted(stats['loss_margin'].items())} if losses else {},
           'second_place_loss_share': stats['second_place_losses'] / losses if losses else None,
           'mean_turns_win': statistics.fmean(stats['turns_win']) if stats['turns_win'] else None,
           'mean_turns_loss': statistics.fmean(stats['turns_loss']) if stats['turns_loss'] else None,
           'mean_points_when_lost': statistics.fmean(stats['points_when_lost']) if stats['points_when_lost'] else None,
           'winner_offset_share': {str(k): v / losses for k, v in sorted(stats['winner_offset'].items())} if losses else {}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-swing-boards', type=int, default=200)
    args = parser.parse_args()
    config, summary = read_header(args.receipt)
    panels = defaultdict(lambda: {'candidate': arm_stats(), 'baseline': arm_stats(),
                                  'control_won_candidate_lost': [], 'candidate_won_control_lost': []})
    count = 0
    for row in iter_games(args.receipt):
        count += 1
        panel = panels[row['panel']]
        for arm in ('candidate', 'baseline'):
            record(panel[arm], row, arm)
        seat = int(row['seat'])
        cand = row['candidate']['winner'] is not None and int(row['candidate']['winner']) == seat
        ctrl = row['baseline']['winner'] is not None and int(row['baseline']['winner']) == seat
        if ctrl and not cand and len(panel['control_won_candidate_lost']) < args.max_swing_boards:
            panel['control_won_candidate_lost'].append({'seed': int(row['seed']), 'seat': seat})
        if cand and not ctrl and len(panel['candidate_won_control_lost']) < args.max_swing_boards:
            panel['candidate_won_control_lost'].append({'seed': int(row['seed']), 'seat': seat})
    report = {'receipt': str(args.receipt), 'rows': count, 'candidate': config.get('candidate'),
              'control': config.get('control'), 'pass': summary.get('pass'), 'panels': {}}
    for name, panel in panels.items():
        report['panels'][name] = {'candidate': finish(panel['candidate']), 'control': finish(panel['baseline']),
                                  'swing_boards': {'control_won_candidate_lost': panel['control_won_candidate_lost'],
                                                   'candidate_won_control_lost': panel['candidate_won_control_lost']}}
    for name, panel in report['panels'].items():
        c, k = panel['candidate'], panel['control']
        print(f"[{name}] candidate {c['win_rate']:.3f} vs control {k['win_rate']:.3f} over {c['games']} games each")
        print(f"  by seat     cand {' '.join(f'{s}:{v:.2f}' for s, v in c['win_rate_by_seat'].items())}")
        print(f"              ctrl {' '.join(f'{s}:{v:.2f}' for s, v in k['win_rate_by_seat'].items())}")
        print(f"  loss margin cand {c['loss_margin_share']}  second-place share {c['second_place_loss_share']:.2f}")
        print(f"  turns       cand win {c['mean_turns_win']:.1f} / loss {c['mean_turns_loss']:.1f}; ctrl win {k['mean_turns_win']:.1f} / loss {k['mean_turns_loss']:.1f}")
        print(f"  who beats the candidate (seat offset): {c['winner_offset_share']}")
        print(f"  swing boards: ctrl-won/cand-lost {len(panel['swing_boards']['control_won_candidate_lost'])}, "
              f"cand-won/ctrl-lost {len(panel['swing_boards']['candidate_won_control_lost'])} (capped)")
    if args.output:
        if args.output.exists():
            raise SystemExit('output exists; choose a new path')
        args.output.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
