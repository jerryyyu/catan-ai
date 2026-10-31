"""Loss attribution, pass 2: replay swing boards and diff the two arms event by event.

Takes a pass-1 report (``scripts/loss_attribution.py``, schema v2), which binds
the receipt's sha256, config, source hashes and checkpoint, and replays each
sampled swing row in both arms
with the exact game construction ``search_eval`` uses (same names, same policy
seeds), and records per-seat trajectories: production received, cards lost to
the robber and to discards, robber moves aimed at each seat, trades accepted,
development cards bought, builds by turn, awards held at the end, and public
points at turns 40 and 60. It then aggregates candidate-seat metrics across
rows where the control won and the candidate lost, versus the reverse, weighting
each row by its pass-1 sampling weight. The aggregates describe what differs on
these boards; they do not by themselves establish what caused the outcome.

Run it against the frozen source snapshot that produced the receipt, because
checkpointed policies refuse a source mismatch:

    PYTHONPATH=runs/economic-parameters-confirmatory-source-v1/src \
    CATAN_PARAMETER_MODEL=runs/economic-parameters-v1/model.json \
    uv run python scripts/replay_swing_boards.py \
        --report runs/loss-attribution-tactical-heldout-v2.json \
        --panel tactical --output runs/swing-replay-tactical-v2.json

The replay refuses to start unless the running source hashes and checkpoint
equal the ones the receipt recorded. Each replayed arm's winner, turn count,
final points and truncation must equal the receipt's terminal record; any
difference is reported as a mismatch with the differing fields and excluded.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.observation import Observation

TERMINAL = ('winner', 'turns', 'points', 'truncated')


def points_at(points_by_turn, cutoff):
    """Public points at the latest observation at or before ``cutoff`` (0 before any)."""
    turns = [t for t in points_by_turn if t <= cutoff]
    return points_by_turn[max(turns)] if turns else 0


def terminal_differences(stored, replayed):
    """Fields of a receipt terminal record that a replay does not reproduce."""
    return {k: {'receipt': stored.get(k), 'replay': replayed.get(k)} for k in TERMINAL if stored.get(k) != replayed.get(k)}


def replay(policy_names, seed, max_turns, watch_seat):
    """Mirror sim.play_game while recording a per-seat trajectory."""
    game = Game.new([f"{name}-{seat}" for seat, name in enumerate(policy_names)], seed)
    policies = [make_policy(name, seed * 17 + seat + 1) for seat, name in enumerate(policy_names)]
    decisions = 0
    track = {'production': Counter(), 'stolen_from': 0, 'stole': 0, 'discarded': 0, 'robber_on_my_tile': 0,
             'trades_accepted_mine': 0, 'trades_accepted_theirs': 0, 'offers_made': 0, 'dev_bought': 0,
             'first_city_turn': None, 'first_fourth_settlement_turn': None, 'builds': Counter(),
             'points_by_turn': {}, 'bank_trades': 0}
    seen_events = 0
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        view = Observation.for_policy(game, seat, policies[seat])
        action = policies[seat].choose(view, seat)
        game.apply(seat, action)
        decisions += 1
        for event in game.observed_events(watch_seat)[seen_events:]:
            kind = event['action'].get('type')
            actor = event['seat']
            deltas = event.get('resource_deltas')
            if kind == 'roll' and event.get('rolled') != 7 and deltas:
                for resource, n in deltas[watch_seat].items():
                    if n > 0:
                        track['production'][resource] += n
            if kind == 'steal':
                if actor == watch_seat:
                    track['stole'] += 1
                elif event['action'].get('victim') == watch_seat:
                    track['stolen_from'] += 1
            if kind == 'discard' and actor == watch_seat:
                track['discarded'] += 1
            if kind == 'move_robber' and actor != watch_seat:
                tile = game.board.tiles[event['action']['tile']]
                me = game.players[watch_seat]
                if any(v in me.settlements or v in me.cities for v in tile.vertices):
                    track['robber_on_my_tile'] += 1
            if kind == 'accept_trade':
                owner = event.get('turn_owner')
                if actor == watch_seat or owner == watch_seat:
                    track['trades_accepted_mine' if owner == watch_seat else 'trades_accepted_theirs'] += 1
            if kind == 'offer_trade' and actor == watch_seat:
                track['offers_made'] += 1
            if kind == 'bank_trade' and actor == watch_seat:
                track['bank_trades'] += 1
            if kind == 'buy_development' and actor == watch_seat:
                track['dev_bought'] += 1
            if actor == watch_seat and kind in ('build_settlement', 'build_city', 'build_road'):
                track['builds'][kind] += 1
                me = game.players[watch_seat]
                if kind == 'build_city' and track['first_city_turn'] is None:
                    track['first_city_turn'] = event['turn']
                if kind == 'build_settlement' and len(me.settlements) + len(me.cities) >= 4 and track['first_fourth_settlement_turn'] is None:
                    track['first_fourth_settlement_turn'] = event['turn']
            track['points_by_turn'][event['turn']] = event['public_points'][watch_seat]
        seen_events = len(game.events)
        if decisions > max_turns * 50:
            break
    me = game.players[watch_seat]
    track['final'] = {'winner': game.winner, 'turns': game.turn, 'points': [p.points for p in game.players],
                      'truncated': game.winner is None,
                      'longest_road': game.longest_road_holder == watch_seat,
                      'largest_army': game.largest_army_holder == watch_seat,
                      'settlements': len(me.settlements), 'cities': len(me.cities), 'roads': len(me.roads),
                      'knights': me.knights, 'hand': sum(me.resources.values()), 'my_points': me.points}
    track['production'] = dict(track['production'])
    track['builds'] = dict(track['builds'])
    track['points_at_turn_40'] = points_at(track['points_by_turn'], 40)
    track['points_at_turn_60'] = points_at(track['points_by_turn'], 60)
    del track['points_by_turn']
    return track


def wmean(pairs):
    """Weighted mean of (value, weight) pairs; None when no weight."""
    pairs = [(v, w) for v, w in pairs if v is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def aggregate(tracks, weights):
    """Weighted candidate-seat means; weights are pass-1 sampling weights."""
    if not tracks:
        return {}
    keys = ['stolen_from', 'stole', 'discarded', 'robber_on_my_tile', 'trades_accepted_mine', 'trades_accepted_theirs',
            'offers_made', 'bank_trades', 'dev_bought', 'points_at_turn_40', 'points_at_turn_60']
    out = {k: wmean((t[k], w) for t, w in zip(tracks, weights)) for k in keys}
    out['production_total'] = wmean((sum(t['production'].values()), w) for t, w in zip(tracks, weights))
    out['first_city_turn'] = wmean((t['first_city_turn'], w) for t, w in zip(tracks, weights))
    out['no_city_share'] = wmean((float(t['first_city_turn'] is None), w) for t, w in zip(tracks, weights))
    out['longest_road_share'] = wmean((float(t['final']['longest_road']), w) for t, w in zip(tracks, weights))
    out['largest_army_share'] = wmean((float(t['final']['largest_army']), w) for t, w in zip(tracks, weights))
    out['final_points'] = wmean((t['final']['my_points'], w) for t, w in zip(tracks, weights))
    return out


def check_identity(report, source, checkpoint):
    """Refuse a report whose producer is not the code and checkpoint running now."""
    if report.get('schema') != 'loss-attribution/v2':
        raise SystemExit('report predates schema v2 (no receipt binding or terminal records); rerun pass 1')
    for key in ('receipt_sha256', 'config', 'source'):
        if not report.get(key):
            raise SystemExit(f'report lacks {key}; rerun pass 1')
    if report['source'] != source:
        changed = sorted(k for k in set(report['source']) | set(source) if report['source'].get(k) != source.get(k))
        raise SystemExit(f'running source differs from the receipt producer: {changed[:8]}')
    if report.get('checkpoint') != checkpoint:
        raise SystemExit('running checkpoint differs from the receipt producer')


def replay_row(config, panel, row, replay_fn=None):
    """Replay one sampled row in both arms; return (arms, differences by arm)."""
    replay_fn = replay_fn or replay
    seed, seat = row['seed'], row['seat']
    arms, differences = {}, {}
    for arm, name in (('candidate', config['candidate']), ('control', config['control'])):
        lineup = list(panel['opponents'])
        lineup.insert(seat, name)
        arms[arm] = replay_fn(lineup, seed, config['max_turns'], seat)
        diff = terminal_differences(row[arm], arms[arm]['final'])
        if diff:
            differences[arm] = diff
    return arms, differences


def run(report, panel_name, replay_fn=None, limit=None, progress=None):
    config = report['config']
    panel = next(p for p in config['panels'] if p['name'] == panel_name)
    results = {}
    mismatches = []
    for direction, block in report['panels'][panel_name]['swing'].items():
        results[direction] = []
        for row in block['selected'][:limit]:
            arms, differences = replay_row(config, panel, row, replay_fn)
            if differences:
                mismatches.append({'seed': row['seed'], 'seat': row['seat'], 'direction': direction,
                                   'differences': differences})
                continue
            results[direction].append({'seed': row['seed'], 'seat': row['seat'], 'weight': row['weight'],
                                       'candidate': arms['candidate'], 'control': arms['control']})
            if progress:
                progress(direction, row)
    summary = {}
    for direction, rows in results.items():
        weights = [r['weight'] for r in rows]
        summary[direction] = {'boards': len(rows),
                              'eligible_rows': report['panels'][panel_name]['swing'][direction]['eligible_rows'],
                              'candidate_seat_in_candidate_arm': aggregate([r['candidate'] for r in rows], weights),
                              'candidate_seat_in_control_arm': aggregate([r['control'] for r in rows], weights)}
    return results, mismatches, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--limit', type=int, help='replay only the first N sampled rows per direction (smoke tests)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit('output exists; choose a new path')
    from catan_ai.experiments import source_files
    from catan_ai.search_eval import model_identity
    report = json.loads(args.report.read_text())
    check_identity(report, source_files(), model_identity(report['config']))
    started = time.monotonic()
    results, mismatches, summary = run(report, args.panel, limit=args.limit, progress=lambda d, r: print(json.dumps(
        {'direction': d, 'seed': r['seed'], 'seat': r['seat'], 'elapsed': round(time.monotonic() - started, 1)}), flush=True))
    out = {'schema': 'swing-replay/v2', 'report': str(args.report), 'receipt': report['receipt'],
           'receipt_sha256': report['receipt_sha256'], 'sampling': report['sampling'], 'panel': args.panel,
           'limit': args.limit, 'source': report['source'], 'checkpoint': report.get('checkpoint'),
           'mismatches': mismatches, 'summary': summary, 'boards': results,
           'elapsed_seconds': time.monotonic() - started}
    for direction, block in summary.items():
        excluded = sum(m['direction'] == direction for m in mismatches)
        print(f"\n== {direction}: {block['boards']} sampled rows of {block['eligible_rows']} eligible "
              f"(replay mismatches excluded: {excluded}); weighted means")
        a, b = block['candidate_seat_in_candidate_arm'], block['candidate_seat_in_control_arm']
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key), b.get(key)
            fa = 'none' if va is None or (isinstance(va, float) and math.isnan(va)) else f'{va:.2f}'
            fb = 'none' if vb is None or (isinstance(vb, float) and math.isnan(vb)) else f'{vb:.2f}'
            print(f"  {key:<26} candidate-arm {fa:>7}   control-arm {fb:>7}")
    if args.output:
        with args.output.open('x') as stream:
            json.dump(out, stream, indent=1)
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
