"""Differential gate for performance-only changes (shengji rule: identical seeded histories).

Plays seeded self-play games of a policy and prints one SHA-256 per game over the
complete action sequence, winner and turn count, plus their combined hash and the
wall time. Run it under the reference source and under the optimized source
(``PYTHONPATH=<snapshot>/src``) and require the combined hashes to be equal.

    PYTHONPATH=<src> CATAN_ENGINE=python .venv/bin/python scripts/trace_hash.py \
        --policy planner_offer_accept --seed 2690000 --games 16
"""
import argparse
import hashlib
import json
import time


def main():
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='planner_offer_accept')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--games', type=int, default=16)
    args = parser.parse_args()
    started = time.monotonic()
    combined = hashlib.sha256()
    for seed in range(args.seed, args.seed + args.games):
        game = Game.new([f'{args.policy}-{i}' for i in range(4)], seed)
        bots = [make_policy(args.policy, seed * 17 + i + 1) for i in range(4)]
        digest = hashlib.sha256()
        while game.winner is None and game.turn < 1000:
            seat = game.current
            action = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
            digest.update(json.dumps([seat, action], sort_keys=True).encode())
            game.apply(seat, action)
        digest.update(json.dumps([game.winner, game.turn]).encode())
        combined.update(digest.digest())
    print(json.dumps({'policy': args.policy, 'seed': args.seed, 'games': args.games,
                      'combined_sha256': combined.hexdigest(), 'seconds': round(time.monotonic() - started, 2)}))


if __name__ == '__main__':
    main()
