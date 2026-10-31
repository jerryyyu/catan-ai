from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

from .bots import make_policy
from .engine import Game
from .observation import Observation


@dataclass(frozen=True)
class GameResult:
    seed: int
    winner: int | None
    turns: int
    points: tuple[int, ...]
    truncated: bool
    search: dict = field(default_factory=dict)
    decision_seconds: tuple[tuple[float, ...], ...] = field(default=(), compare=False)
    step_seconds: tuple[tuple[float, ...], ...] = field(default=(), compare=False)
    turn_timings: tuple[dict, ...] = field(default=(), compare=False)
    policy_metadata: dict = field(default_factory=dict)
    external: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "winner": self.winner,
            "turns": self.turns,
            "points": self.points,
            "truncated": self.truncated,
            "search": self.search,
            "decision_seconds": self.decision_seconds,
            "step_seconds": self.step_seconds,
            "turn_timings": self.turn_timings,
            "policy_metadata": self.policy_metadata,
            "external": self.external,
        }


def play_game(policy_names: list[str], seed: int, max_turns: int = 2000) -> GameResult:
    if len(policy_names) != 4:
        raise ValueError("four policy names are required")
    game = Game.new([f"{name}-{seat}" for seat, name in enumerate(policy_names)], seed)
    policies = [make_policy(name, seed * 17 + seat + 1) for seat, name in enumerate(policy_names)]
    decisions = 0
    decision_seconds = [[] for _ in range(4)]
    step_seconds = [[] for _ in range(4)]
    turn_timings = []
    turn_started = None
    turn_owner = None
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        step_started = time.perf_counter()
        if turn_started is None and not game.phase.startswith('setup'):
            turn_started, turn_owner = step_started, seat
        turn = game.turn
        view = Observation.for_policy(game, seat, policies[seat])
        started = time.perf_counter()
        action = policies[seat].choose(view, seat)
        decision_seconds[seat].append(time.perf_counter() - started)
        game.apply(seat, action)
        step_finished = time.perf_counter()
        step_seconds[seat].append(step_finished - step_started)
        if turn_started is not None and (game.turn != turn or game.winner is not None):
            turn_timings.append({'turn': turn, 'owner': turn_owner,
                                 'seconds': step_finished - turn_started, 'complete': True})
            turn_started = None
        decisions += 1
        if decisions > max_turns * 50:
            break
    if turn_started is not None:
        turn_timings.append({'turn': game.turn, 'owner': turn_owner,
                             'seconds': step_finished - turn_started, 'complete': False})
    return GameResult(
        seed=seed,
        winner=game.winner,
        turns=game.turn,
        points=tuple(p.points for p in game.players),
        truncated=game.winner is None,
        search={str(s): p.diagnostics for s, p in enumerate(policies)
                if hasattr(p, 'diagnostics') and not hasattr(p, 'external_diagnostics')},
        decision_seconds=tuple(tuple(t) for t in decision_seconds),
        step_seconds=tuple(tuple(t) for t in step_seconds),
        turn_timings=tuple(turn_timings),
        policy_metadata={str(s): {'model_sha256': p.model_hash, 'load_error': p.load_error}
                         for s, p in enumerate(policies) if hasattr(p, 'model_hash')},
        external={str(s): {'identity': p.identity, 'diagnostics': p.external_diagnostics}
                  for s, p in enumerate(policies) if hasattr(p, 'external_diagnostics')},
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
