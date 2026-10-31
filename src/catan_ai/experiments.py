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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_files() -> dict[str, str]:
    from .engine import BACKEND, NATIVE_BINARY
    root = Path(__file__).parent
    result = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    result['engine_backend'] = hashlib.sha256(BACKEND.encode()).hexdigest()
    if NATIVE_BINARY:
        result['engine_binary'] = hashlib.sha256(Path(NATIVE_BINARY).read_bytes()).hexdigest()
        from . import _scoring_cy
        result['scoring_binary'] = hashlib.sha256(Path(_scoring_cy.__file__).read_bytes()).hexdigest()
    return result


def cluster_interval(rows: list[dict]) -> tuple[float, float] | None:
    """Hoeffding interval for independent seed clusters bounded in [0, 1]."""
    clusters: dict[int, list[dict]] = {}
    for row in rows:
        if row['truncated']:
            return None
        clusters.setdefault(row['seed'], []).append(row)
    if not clusters:
        return None
    values = []
    for cluster in clusters.values():
        if len(cluster) != 4 or {r['candidate_seat'] for r in cluster} != set(range(4)):
            raise ValueError('each seed requires exactly four distinct candidate seats')
        values.append(sum(r['won'] for r in cluster) / 4)
    mean = sum(values) / len(values)
    margin = math.sqrt(math.log(40) / (2 * len(values)))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def run(config: dict) -> dict:
    started = time.monotonic()
    candidate = config["candidate"]
    control = config["control"]
    games_per_seat = int(config["games_per_seat"])
    base_seed = int(config["seed"])
    max_turns = int(config.get("max_turns", 2000))
    if games_per_seat <= 0 or max_turns <= 0:
        raise ValueError('games_per_seat and max_turns must be positive')
    bar = config.get('pass_bar', {})
    if bar.get('metric') != 'win_rate' or not 0 <= float(bar['minimum']) <= 1:
        raise ValueError('a win_rate pass bar between 0 and 1 is required')
    source_before = source_files()
    rows = []
    wins = 0
    completed = 0
    for candidate_seat in range(4):
        policies = [control] * 4
        policies[candidate_seat] = candidate
        for offset in range(games_per_seat):
            seed = base_seed + offset
            result = play_game(policies, seed, max_turns)
            if result.truncated != (result.winner is None) or (result.winner is not None and result.winner not in range(4)):
                raise ValueError('inconsistent terminal outcome')
            won = result.winner == candidate_seat
            if not result.truncated:
                completed += 1
                wins += int(won)
            rows.append({"candidate_seat": candidate_seat, "won": won, **result.as_dict()})
    interval = cluster_interval(rows)
    minimum = float(config.get("pass_bar", {}).get("minimum", 0.0))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": 2,
        "name": config["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_revision": source_revision(),
        "source_files_sha256": source_before,
        "source_unchanged": source_before == source_files(),
        "config": config,
        "summary": {
            "scheduled": len(rows),
            "completed": completed,
            "truncated": len(rows) - completed,
            "candidate_wins": wins,
            "win_rate": wins / len(rows) if completed == len(rows) else None,
            "seed_cluster_95": interval,
            "interval_method": "two-sided Hoeffding bound on independent seed averages",
            "pass": interval is not None and interval[0] >= minimum and source_before == source_files(),
            "pass_rule": f"all games complete, source unchanged, seed-cluster lower bound >= {minimum}",
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.output_dir / f"{config['name']}-{stamp}.json"
    with path.open('x') as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
