from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .sim import play_game


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def source_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config: dict) -> dict:
    started = time.monotonic()
    candidate = config["candidate"]
    control = config["control"]
    games_per_seat = int(config["games_per_seat"])
    base_seed = int(config["seed"])
    max_turns = int(config.get("max_turns", 2000))
    rows = []
    wins = 0
    completed = 0
    for candidate_seat in range(4):
        policies = [control] * 4
        policies[candidate_seat] = candidate
        for offset in range(games_per_seat):
            seed = base_seed + offset
            result = play_game(policies, seed, max_turns)
            won = result.winner == candidate_seat
            if not result.truncated:
                completed += 1
                wins += int(won)
            rows.append({"candidate_seat": candidate_seat, "won": won, **result.as_dict()})
    low, high = wilson(wins, completed)
    minimum = float(config.get("pass_bar", {}).get("minimum", 0.0))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": 1,
        "name": config["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_revision": source_revision(),
        "config": config,
        "summary": {
            "scheduled": len(rows),
            "completed": completed,
            "truncated": len(rows) - completed,
            "candidate_wins": wins,
            "win_rate": wins / completed if completed else 0.0,
            "wilson_95": [low, high],
            "pass": completed == len(rows) and low >= minimum,
            "pass_rule": f"all games complete and Wilson lower bound >= {minimum}",
            "duration_seconds": time.monotonic() - started,
        },
        "games": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired, seat-rotated bot experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"{config['name']}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
