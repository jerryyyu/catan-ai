from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .bots import make_policy
from .engine import Game


@dataclass(frozen=True)
class GameResult:
    seed: int
    winner: int | None
    turns: int
    points: tuple[int, ...]
    truncated: bool

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "winner": self.winner,
            "turns": self.turns,
            "points": self.points,
            "truncated": self.truncated,
        }


def play_game(policy_names: list[str], seed: int, max_turns: int = 2000) -> GameResult:
    if len(policy_names) != 4:
        raise ValueError("four policy names are required")
    game = Game.new([f"{name}-{seat}" for seat, name in enumerate(policy_names)], seed)
    policies = [make_policy(name, seed * 17 + seat + 1) for seat, name in enumerate(policy_names)]
    decisions = 0
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        game.apply(seat, policies[seat].choose(game, seat))
        decisions += 1
        if decisions > max_turns * 50:
            break
    return GameResult(
        seed=seed,
        winner=game.winner,
        turns=game.turn,
        points=tuple(p.points for p in game.players),
        truncated=game.winner is None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Catan AI simulations")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policies", nargs=4, default=["heuristic", "random", "random", "random"])
    parser.add_argument("--max-turns", type=int, default=2000)
    args = parser.parse_args()
    results = [play_game(args.policies, args.seed + i, args.max_turns) for i in range(args.games)]
    wins = [sum(r.winner == seat for r in results) for seat in range(4)]
    print(json.dumps({"games": args.games, "wins": wins, "truncated": sum(r.truncated for r in results)}, indent=2))


if __name__ == "__main__":
    main()
