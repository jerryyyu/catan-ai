"""Loss attribution, pass 2: replay swing boards and diff the two arms event by event.

Takes the pass-1 report (``scripts/loss_attribution.py``) and the evaluator
config that produced the receipt, replays each listed swing board in both arms
with the exact game construction ``search_eval`` uses (same names, same policy
seeds), and records per-seat trajectories: production received, cards lost to
the robber and to discards, robber moves aimed at each seat, trades accepted,
development cards bought, builds by turn, awards held at the end, and public
points per turn. It then aggregates candidate-seat metrics across boards where
the control won and the candidate lost, versus the reverse.

Run it against the frozen source snapshot that produced the receipt, because
checkpointed policies refuse a source mismatch:

    PYTHONPATH=runs/economic-parameters-confirmatory-source-v1/src \
    CATAN_PARAMETER_MODEL=runs/economic-parameters-v1/model.json \
    uv run python scripts/replay_swing_boards.py \
        --report runs/loss-attribution-tactical-heldout-v1.json \
        --config runs/economic-parameters-confirmatory-source-v1/experiments/economic-parameters-tactical-heldout.json \
        --panel tactical --limit 50 --output runs/swing-replay-tactical-v1.json

Replays confirm determinism: a replayed terminal outcome that differs from the
receipt's row is reported as a mismatch and excluded, never silently kept.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.observation import Observation


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
                      'longest_road': game.longest_road_holder == watch_seat,
                      'largest_army': game.largest_army_holder == watch_seat,
                      'settlements': len(me.settlements), 'cities': len(me.cities), 'roads': len(me.roads),
                      'knights': me.knights, 'hand': sum(me.resources.values()), 'my_points': me.points}
    track['production'] = dict(track['production'])
    track['builds'] = dict(track['builds'])
    track['points_at_turn_40'] = max((p for t, p in track['points_by_turn'].items() if t <= 40), default=0)
    track['points_at_turn_60'] = max((p for t, p in track['points_by_turn'].items() if t <= 60), default=0)
    del track['points_by_turn']
    return track


def aggregate(tracks):
    keys = ['stolen_from', 'stole', 'discarded', 'robber_on_my_tile', 'trades_accepted_mine', 'trades_accepted_theirs',
            'offers_made', 'bank_trades', 'dev_bought', 'points_at_turn_40', 'points_at_turn_60']
    out = {k: statistics.fmean(t[k] for t in tracks) for k in keys} if tracks else {}
    if tracks:
        out['production_total'] = statistics.fmean(sum(t['production'].values()) for t in tracks)
        out['first_city_turn'] = statistics.fmean(t['first_city_turn'] for t in tracks if t['first_city_turn'] is not None) \
            if any(t['first_city_turn'] is not None for t in tracks) else None
        out['no_city_share'] = sum(t['first_city_turn'] is None for t in tracks) / len(tracks)
        out['longest_road_share'] = sum(t['final']['longest_road'] for t in tracks) / len(tracks)
        out['largest_army_share'] = sum(t['final']['largest_army'] for t in tracks) / len(tracks)
        out['final_points'] = statistics.fmean(t['final']['my_points'] for t in tracks)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--limit', type=int, default=50, help='boards per direction')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    config = json.loads(args.config.read_text())
    panel = next(p for p in config['panels'] if p['name'] == args.panel)
    swing = report['panels'][args.panel]['swing_boards']
    results = {'control_won_candidate_lost': [], 'candidate_won_control_lost': []}
    mismatches = []
    started = time.monotonic()
    for direction, boards in swing.items():
        for board in boards[:args.limit]:
            seed, seat = board['seed'], board['seat']
            arms = {}
            for arm, name in (('candidate', config['candidate']), ('control', config['control'])):
                lineup = list(panel['opponents'])
                lineup.insert(seat, name)
                arms[arm] = replay(lineup, seed, config['max_turns'], seat)
            cand_won = arms['candidate']['final']['winner'] == seat
            ctrl_won = arms['control']['final']['winner'] == seat
            expected = direction == 'candidate_won_control_lost'
            if (cand_won, ctrl_won) != ((True, False) if expected else (False, True)):
                mismatches.append({'seed': seed, 'seat': seat, 'direction': direction,
                                   'replayed': {'candidate_won': cand_won, 'control_won': ctrl_won}})
                continue
            results[direction].append({'seed': seed, 'seat': seat, 'candidate': arms['candidate'], 'control': arms['control']})
            print(json.dumps({'direction': direction, 'seed': seed, 'seat': seat,
                              'elapsed': round(time.monotonic() - started, 1)}), flush=True)
    summary = {}
    for direction, rows in results.items():
        summary[direction] = {'boards': len(rows),
                              'candidate_seat_in_candidate_arm': aggregate([r['candidate'] for r in rows]),
                              'candidate_seat_in_control_arm': aggregate([r['control'] for r in rows])}
    out = {'report': str(args.report), 'config': str(args.config), 'panel': args.panel, 'limit': args.limit,
           'mismatches': mismatches, 'summary': summary, 'boards': results,
           'elapsed_seconds': time.monotonic() - started}
    for direction, block in summary.items():
        print(f"\n== {direction}: {block['boards']} boards (replay mismatches excluded: {sum(m['direction'] == direction for m in mismatches)})")
        a, b = block['candidate_seat_in_candidate_arm'], block['candidate_seat_in_control_arm']
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key), b.get(key)
            fa = 'none' if va is None else f'{va:.2f}'
            fb = 'none' if vb is None else f'{vb:.2f}'
            print(f"  {key:<26} candidate-arm {fa:>7}   control-arm {fb:>7}")
    if args.output:
        if args.output.exists():
            raise SystemExit('output exists; choose a new path')
        args.output.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
