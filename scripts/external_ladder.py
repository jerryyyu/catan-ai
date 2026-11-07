"""External ladder: our bot alone against three Catanatron players of one variant.

Descriptive comparison (Jerry 2026-11-07: "try play vs stronger"; "retire when we beat
them too easily"). For each seed the candidate takes each seat once against three
copies of the variant (``ExternalSearchBot``, belief-sampled hidden state, no cheating).
Per game: winner, turns, external fallbacks/budget failures, decision times. The
fair-share win rate is .25.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import subprocess
import time


def play(job):
    seed, seat, variant, candidate, budget, max_turns, path = job
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.external_policy import ExternalSearchBot
    from catan_ai.observation import Observation
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    game = Game.new([f'p{i}' for i in range(4)], seed)
    bots = [make_policy(candidate, seed * 17 + i + 1) if i == seat else
            ExternalSearchBot(seed * 1000 + i, seconds=budget['decision'], turn_seconds=budget['turn'], variant=variant)
            for i in range(4)]
    try:
        while game.winner is None and game.turn < max_turns:
            s = game.current
            game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    finally:
        for i, b in enumerate(bots):
            if i != seat:
                b.close()
    ext = [d for i, b in enumerate(bots) if i != seat for d in b.diagnostics]
    record = {'seed': seed, 'seat': seat, 'variant': variant, 'winner': game.winner, 'turns': game.turn,
              'truncated': game.winner is None, 'seconds': time.monotonic() - started,
              'external_decisions': len(ext), 'external_fallbacks': sum(d['fallback'] for d in ext),
              'external_budget_failures': sum(d['budget_failure'] for d in ext),
              'external_errors': sorted({d['error'] for d in ext if d['error']})[:5],
              'external_seconds_p95': sorted(d['seconds'] for d in ext)[int(.95 * len(ext))] if ext else None,
              'identity': next((b.identity for i, b in enumerate(bots) if i != seat and b.identity), None)}
    with path.open('x') as stream:
        json.dump(record, stream)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--variants', nargs='*', help='subset of the config variants (default all)')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    meta = args.root / 'run.json'
    if not meta.exists():
        meta.write_text(json.dumps({'config': cfg, 'host': platform.node(),
                                    'revision': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                                    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1))
    jobs = []
    for variant in args.variants or cfg['variants']:
        spec = cfg['variants'][variant]
        (args.root / variant).mkdir(exist_ok=True)
        jobs += [(seed, seat, variant, cfg['candidate'], spec['budget'], cfg['max_turns'],
                  str(args.root / variant / f'{seed}-{seat}.json'))
                 for seed in range(spec['seed'], spec['seed'] + spec['seeds']) for seat in range(4)]
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(play, jobs):
            done += 1
            print(json.dumps({'done': done, 'of': len(jobs), 'variant': r['variant'], 'won': r['winner'] == r['seat'],
                              'fallbacks': r['external_fallbacks'], 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
