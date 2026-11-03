"""Offer behaviour of trading planners: counts, acceptance, trades to near-winners, estimate quality.

Diagnostic only, not a strength claim. Each seed is played with the candidate rotated
through all four seats against the table's lineup. True hands are read from the engine
to grade the card-counted estimate after the fact; they are never a policy input.
"""
import argparse
import json
from pathlib import Path
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation

TABLES = {
    'tactical': ['tactical'] * 3,
    'mixed': ['builder', 'tactical', 'expander'],
    'self': ['planner_unified_offer'] * 3,
    'strict': ['planner_strict_acceptor'] * 3,
}
NEAR_WIN = 8


def play(candidate, opponents, seed, seat, max_turns):
    names = list(opponents)
    names.insert(seat, candidate)
    game = Game.new([f'{n}-{i}' for i, n in enumerate(names)], seed)
    policies = [make_policy(n, seed * 17 + i + 1) for i, n in enumerate(names)]
    stats = {'offers': 0, 'accepted': 0, 'offers_to_near_win': 0, 'accepted_to_near_win': 0,
             'recipient_lacked_card': 0, 'own_turns': 0}
    pending = None
    while game.winner is None and game.turn < max_turns:
        current = game.current
        view = Observation.for_policy(game, current, policies[current])
        action = policies[current].choose(view, current)
        if current == seat and action['type'] == 'end_turn':
            stats['own_turns'] += 1
        if current == seat and action['type'] == 'offer_trade':
            recipient = game.players[action['recipient']]
            near = recipient.public_points >= NEAR_WIN
            (card, count), = action['receive'].items()
            stats['offers'] += 1
            stats['offers_to_near_win'] += near
            stats['recipient_lacked_card'] += recipient.resources[Resource(card)] < count
            pending = near
        if pending is not None and action['type'] in ('accept_trade', 'reject_trade'):
            if action['type'] == 'accept_trade':
                stats['accepted'] += 1
                stats['accepted_to_near_win'] += pending
            pending = None
        game.apply(current, action)
    stats['won'] = game.winner == seat
    stats['truncated'] = game.winner is None
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--tables', nargs='+', default=list(TABLES))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--max-turns', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    rows = []
    for candidate in args.candidates:
        for t, table in enumerate(args.tables):
            total = {}
            for i in range(args.seeds):
                for seat in range(4):
                    for key, value in play(candidate, TABLES[table], args.seed + 1000 * t + i, seat,
                                           args.max_turns).items():
                        total[key] = total.get(key, 0) + value
            games = 4 * args.seeds
            rows.append({'candidate': candidate, 'table': table, 'games': games, **total,
                         'offers_per_game': total['offers'] / games,
                         'accept_rate': total['accepted'] / total['offers'] if total['offers'] else None,
                         'accepted_to_near_win_per_100_games': 100 * total['accepted_to_near_win'] / games,
                         'win_rate': total['won'] / games})
            print(json.dumps(rows[-1]), flush=True)
    result = {'source': source_files(), 'seed': args.seed, 'seeds': args.seeds, 'tables': args.tables,
              'near_win_points': NEAR_WIN, 'seconds': time.monotonic() - started, 'rows': rows,
              'scope': 'diagnostic: offer behaviour, not a strength claim (unpaired win rates)'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
