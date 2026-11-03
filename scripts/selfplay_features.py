"""Which early-game features predict the winner when four copies of one bot play?

Diagnostic for board row 2 (loss attribution), not a strength claim. In a mirror
game every seat runs the same policy, so differences in outcome come from seat
order, board luck and the positions the shared policy steers each seat into.
If the seat with the best opening (production pips, resource coverage) wins far
more often than 1 in 4, opening placement is a lever worth a powered screen;
if seat order or later features dominate, look elsewhere.

Features use public engine state only, read after the game ends (never fed to a
policy):
- setup: expected production pips of the two opening settlements (dice pips,
  robber-free), distinct resources produced, ore+grain pips, brick+lumber pips,
  port access; and each seat's rank on setup pips among the four (1 = highest).
- midgame: public points at turn 40 and 60 (rank among the four).

Usage (run against a frozen snapshot):
    PYTHONPATH=runs/<snapshot>/src CATAN_ENGINE=python \
      .venv/bin/python <worktree>/scripts/selfplay_features.py --policy planner_offer_accept \
      --seed 2480000 --games 256 --output runs/selfplay-features-v1.json
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


def pips(number):
    return 0 if number is None else 6 - abs(7 - number)


def setup_features(game, seat):
    vertices = game.players[seat].settlements | game.players[seat].cities
    income = defaultdict(int)
    for tile in game.board.tiles:
        if isinstance(tile.resource, Resource):
            for v in tile.vertices:
                if v in vertices:
                    income[tile.resource] += pips(tile.number)
    port = any(set(vs) & vertices for vs, _ in game.board.ports)
    return {'pips': sum(income.values()), 'diversity': sum(v > 0 for v in income.values()),
            'ore_grain': income[Resource.ORE] + income[Resource.GRAIN],
            'brick_lumber': income[Resource.BRICK] + income[Resource.LUMBER], 'port': port}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0] * len(values)
    for r, i in enumerate(order):
        out[i] = 1 + sum(values[j] > values[i] for j in range(len(values)))
    return out


def play(policy, seed, max_turns):
    names = [f'{policy}-{i}' for i in range(4)]
    game = Game.new(names, seed)
    bots = [make_policy(policy, seed * 17 + i + 1) for i in range(4)]
    setup, points = None, {}
    while game.winner is None and game.turn < max_turns:
        if setup is None and not game.phase.startswith('setup'):
            setup = [setup_features(game, s) for s in range(4)]
        for mark in (40, 60):
            if mark not in points and game.turn >= mark:
                points[mark] = [p.public_points for p in game.players]
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    rows = []
    pip_rank = ranks([f['pips'] for f in setup])
    for s in range(4):
        row = {'seed': seed, 'seat': s, 'won': game.winner == s, **setup[s], 'pips_rank': pip_rank[s]}
        for mark, values in points.items():
            row[f'points_{mark}'] = values[s]
            row[f'points_{mark}_rank'] = ranks(values)[s]
        rows.append(row)
    return rows, game.winner is None, game.turn


def summarise(rows):
    def rate(pred):
        sel = [r for r in rows if pred(r)]
        return {'n': len(sel), 'win_rate': sum(r['won'] for r in sel) / len(sel) if sel else None}
    out = {'by_seat': {s: rate(lambda r, s=s: r['seat'] == s) for s in range(4)},
           'by_setup_pips_rank': {k: rate(lambda r, k=k: r['pips_rank'] == k) for k in range(1, 5)},
           'by_diversity': {k: rate(lambda r, k=k: r['diversity'] == k) for k in range(1, 6)},
           'by_port': {str(k): rate(lambda r, k=k: r['port'] == k) for k in (True, False)}}
    for mark in (40, 60):
        key = f'points_{mark}_rank'
        out[f'by_{key}'] = {k: rate(lambda r, k=k, key=key: r.get(key) == k) for k in range(1, 5)}
    winners = [r for r in rows if r['won']]
    losers = [r for r in rows if not r['won']]
    out['mean_feature_winner_vs_loser'] = {
        f: [sum(r[f] for r in winners) / len(winners), sum(r[f] for r in losers) / len(losers)]
        for f in ('pips', 'diversity', 'ore_grain', 'brick_lumber')} if winners and losers else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--games', type=int, default=256)
    parser.add_argument('--max-turns', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    rows, truncated = [], 0
    for i in range(args.games):
        game_rows, trunc, _ = play(args.policy, args.seed + i, args.max_turns)
        truncated += trunc
        rows.extend(game_rows)
        if (i + 1) % 32 == 0:
            print(json.dumps({'games': i + 1, 'seconds': round(time.monotonic() - started)}), flush=True)
    complete = [r for r in rows if any(x['won'] for x in rows if x['seed'] == r['seed'])]
    result = {'scope': 'diagnostic: mirror-game feature/outcome association, not a strength claim',
              'policy': args.policy, 'seed': args.seed, 'games': args.games, 'truncated': truncated,
              'source': source_files(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds': time.monotonic() - started, 'summary': summarise(complete), 'rows': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['summary'], indent=1))


if __name__ == '__main__':
    main()
