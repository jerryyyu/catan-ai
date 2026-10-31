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

Every count above is descriptive. Terminal margins say how far behind a loss
ended, not which decision caused it or what a different policy could recover.

Swing rows for pass 2 are a declared seeded sample, not a file-order prefix:
eligible rows (both arms complete, no truncation, no external-comparator fault)
are grouped by board seed, ``--sample-clusters`` seeds are drawn uniformly
without replacement per panel and direction, and one eligible seat is drawn
within each. Each selected row carries ``weight`` = eligible seats on its board,
so weighted means estimate the row-level mean over all eligible swing rows.
Excluded rows are counted, never silently dropped. The report binds the
receipt's sha256, config, source hashes and checkpoint, and stores both arms'
terminal records so pass 2 can verify its replays.

Usage:
    uv run python scripts/loss_attribution.py runs/economic-parameters-tactical-heldout-v1.json \
        --sample-seed 20261031 --output runs/loss-attribution-tactical-heldout-v2.json
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

SCHEMA = 'loss-attribution/v2'
TERMINAL = ('winner', 'turns', 'points', 'truncated')

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
    """Top-level identity fields (config, summary, source, checkpoint) without the games array."""
    if ijson is None:
        data = json.loads(path.read_text())
        return {k: data.get(k) for k in ('config', 'summary', 'source', 'checkpoint')}
    header = {}
    for key in ('config', 'summary', 'source', 'checkpoint'):
        with path.open('rb') as stream:
            header[key] = next(ijson.items(stream, key, use_float=True), None)
    return header


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def terminal(result):
    return {'winner': None if result['winner'] is None else int(result['winner']), 'turns': int(result['turns']),
            'points': [int(p) for p in result['points']], 'truncated': bool(result['truncated'])}


def row_faults(row):
    """External-comparator faults in a row, by the evaluator's own fault definition."""
    faults = 0
    for arm in ('candidate', 'baseline'):
        result = row[arm]
        for seat, evidence in (result.get('external') or {}).items():
            records = evidence.get('diagnostics', [])
            faults += sum(bool(d.get('fallback') or d.get('budget_failure')) for d in records)
            faults += sum(t > 2 for t in (result.get('step_seconds') or [[], [], [], []])[int(seat)])
            per_turn = Counter()
            for d in records:
                if not d['phase'].startswith('setup'):
                    per_turn[d['turn']] += d['seconds']
            faults += sum(t > 10 for t in per_turn.values())
    return faults


def row_status(row):
    for arm in ('candidate', 'baseline'):
        result = row[arm]
        if result['truncated'] or result['winner'] is None or int(result['winner']) not in range(4):
            return 'truncated'
    return 'faulted' if row_faults(row) else 'complete'


def sample_swing(rows, clusters, rng):
    """Uniform seeded draw of board seeds, then one eligible seat per drawn board."""
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row)
    seeds = sorted(by_seed)
    chosen = sorted(rng.sample(seeds, min(clusters, len(seeds))))
    selected = []
    for seed in chosen:
        options = sorted(by_seed[seed], key=lambda r: r['seat'])
        pick = options[rng.randrange(len(options))]
        selected.append({**pick, 'weight': len(options)})
    return selected


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
           'loss_margin_counts': dict(sorted(stats['loss_margin'].items())),
           'loss_margin_share': {k: v / losses for k, v in sorted(stats['loss_margin'].items())} if losses else {},
           'second_place_loss_share': stats['second_place_losses'] / losses if losses else None,
           'mean_turns_win': statistics.fmean(stats['turns_win']) if stats['turns_win'] else None,
           'mean_turns_loss': statistics.fmean(stats['turns_loss']) if stats['turns_loss'] else None,
           'mean_points_when_lost': statistics.fmean(stats['points_when_lost']) if stats['points_when_lost'] else None,
           'winner_offset_share': {str(k): v / losses for k, v in sorted(stats['winner_offset'].items())} if losses else {}}
    return out


def fmt(value, spec='.3f'):
    return 'n/a' if value is None else format(value, spec)


def describe(name, panel):
    c, k = panel['candidate'], panel['control']
    lines = [f"[{name}] candidate {fmt(c['win_rate'])} vs control {fmt(k['win_rate'])} over {c['games']} games each "
             f"(excluded rows: {panel['excluded']})",
             f"  by seat     cand {' '.join(f'{s}:{v:.2f}' for s, v in c['win_rate_by_seat'].items())}",
             f"              ctrl {' '.join(f'{s}:{v:.2f}' for s, v in k['win_rate_by_seat'].items())}",
             f"  loss margin cand {c['loss_margin_share']}  second-place share {fmt(c['second_place_loss_share'], '.2f')}",
             f"  turns       cand win {fmt(c['mean_turns_win'], '.1f')} / loss {fmt(c['mean_turns_loss'], '.1f')}; "
             f"ctrl win {fmt(k['mean_turns_win'], '.1f')} / loss {fmt(k['mean_turns_loss'], '.1f')}",
             f"  who beats the candidate (seat offset): {c['winner_offset_share']}"]
    for direction, block in panel['swing'].items():
        lines.append(f"  {direction}: {block['eligible_rows']} eligible rows on {block['eligible_boards']} boards; "
                     f"sampled {len(block['selected'])}")
    return '\n'.join(lines)


def analyse(rows, sample_clusters, sample_seed):
    panels = defaultdict(lambda: {'candidate': arm_stats(), 'baseline': arm_stats(), 'excluded': Counter(),
                                  'control_won_candidate_lost': [], 'candidate_won_control_lost': []})
    count = 0
    for row in rows:
        count += 1
        panel = panels[row['panel']]
        for arm in ('candidate', 'baseline'):
            record(panel[arm], row, arm)
        status = row_status(row)
        if status != 'complete':
            panel['excluded'][status] += 1
            continue
        seat = int(row['seat'])
        cand = int(row['candidate']['winner']) == seat
        ctrl = int(row['baseline']['winner']) == seat
        if cand == ctrl:
            continue
        entry = {'seed': int(row['seed']), 'seat': seat,
                 'candidate': terminal(row['candidate']), 'control': terminal(row['baseline'])}
        panel['candidate_won_control_lost' if cand else 'control_won_candidate_lost'].append(entry)
    report = {}
    for name in sorted(panels):
        panel = panels[name]
        swing = {}
        for direction in ('control_won_candidate_lost', 'candidate_won_control_lost'):
            eligible = panel[direction]
            rng = random.Random(f'{sample_seed}:{name}:{direction}')
            swing[direction] = {'eligible_rows': len(eligible), 'eligible_boards': len({r['seed'] for r in eligible}),
                                'selected': sample_swing(eligible, sample_clusters, rng)}
        report[name] = {'candidate': finish(panel['candidate']), 'control': finish(panel['baseline']),
                        'excluded': dict(panel['excluded']), 'swing': swing}
    return count, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-seed', type=int, required=True, help='declared seed for the swing-board draw')
    parser.add_argument('--sample-clusters', type=int, default=100, help='board seeds per panel and direction')
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit('output exists; choose a new path')
    header = read_header(args.receipt)
    config = header['config'] if isinstance(header['config'], dict) else {}
    count, panels = analyse(iter_games(args.receipt), args.sample_clusters, args.sample_seed)
    report = {'schema': SCHEMA, 'receipt': str(args.receipt), 'receipt_sha256': file_sha256(args.receipt),
              'config': config, 'source': header['source'], 'checkpoint': header['checkpoint'],
              'rows': count, 'candidate': config.get('candidate'), 'control': config.get('control'),
              'pass': (header['summary'] or {}).get('pass'),
              'sampling': {'method': 'uniform board seeds without replacement, one uniform eligible seat per board; '
                                     'weight = eligible seats on the board',
                           'seed': args.sample_seed, 'clusters_per_direction': args.sample_clusters},
              'panels': panels}
    for name, panel in panels.items():
        print(describe(name, panel))
    if args.output:
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
