"""Phase diagnostic: headroom at non-build decisions (Jerry 2026-11-06; evaluation, not training data).

Same method as ``search_diagnostic.py`` (paired playouts from the TRUE game, the served bot
finishing our seat's game, per-playout bits kept for split-half analysis), applied to the
decisions that diagnostic did not cover:

- ``setup_settlement``: opening settlement placement (every legal vertex); one of seat 0's
  two placements per game, alternating by seed parity;
- ``robber``: robber tile after a 7 or a knight (every legal tile; the served bot steals);
- ``trade_response``: accept or reject an opponent's offer.

Seat 0 plays the PV baseline (``planner_pv2_b_fix``), which delegates these phases to the
served bot, so the served move is also the realized move. Analysis reuses
``analyze_search_diagnostic.analyze`` per phase (all options are kind ``first``).
"""
import argparse
from copy import deepcopy
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_diagnostic import run_playout  # noqa: E402


def run_game(args):
    seed, cfg, lineup_name, path = args
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    path = Path(path)
    if path.exists():
        return 'exists'
    started = time.monotonic()
    opponents = cfg['lineups'][lineup_name]
    names = ['planner_pv2_b_fix'] + opponents
    game = Game.new([f'{n}-{i}' for i, n in enumerate(names)], seed)
    bots = [make_policy(n, seed * 17 + i + 1) for i, n in enumerate(names)]
    shadow = make_policy('planner_offer_accept', seed * 17 + 1)
    playout_cls = ['planner_offer_accept'] + opponents
    setup_index, setups, counts, decisions = seed % 2, 0, {}, []
    while game.winner is None and game.turn < cfg['max_turns']:
        s = game.current
        view = Observation.for_policy(game, s, bots[s])
        phase = game.phase
        probe = False
        if s == 0 and phase in cfg['phases'] and len(view.actions) > 1:
            if phase == 'setup_settlement':
                probe = setups == setup_index
                setups += 1
            else:
                probe = counts.get(phase, 0) < cfg['max_per_game'][phase]
        if probe:
            counts[phase] = counts.get(phase, 0) + 1
            shadow.last_offer_turn = bots[0].last_offer_turn
            served = deepcopy(shadow).choose(view, s)
            n = cfg['playouts'][phase]
            rows = []
            for opt in view.actions:
                bits = []
                for r in range(n):
                    w = run_playout(game, s, playout_cls, seed * 31, opt, 'first',
                                    seed * 1_000_003 + game.turn * 1009 + len(decisions) * 100_003 + r, cfg['max_turns'])
                    if w is None:
                        bits = None
                        break
                    bits.append(w)
                if bits is not None:
                    rows.append({'kind': 'first', 'option': opt, 'bits': bits})
            action = bots[0].choose(view, s)
            decisions.append({'phase': phase, 'turn': game.turn, 'points': [p.public_points for p in game.players],
                              'served': served, 'pv': action, 'options': rows})
            game.apply(s, action)
            continue
        game.apply(s, bots[s].choose(view, s))
    record = {'seed': seed, 'lineup': lineup_name, 'winner': game.winner, 'turns': game.turn,
              'seconds': time.monotonic() - started, 'decisions': decisions}
    with path.open('x') as stream:
        json.dump(record, stream)
    return 'done'


def analyze_root(root, output):
    from analyze_search_diagnostic import analyze
    rng = random.Random(31)
    result = {}
    for lineup_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        by_phase = {}
        for f in sorted(lineup_dir.glob('*.json')):
            rec = json.loads(f.read_text())
            for d in rec['decisions']:
                by_phase.setdefault(d['phase'], []).append((rec['seed'], d))
        result[lineup_dir.name] = {}
        for phase, decisions in sorted(by_phase.items()):
            out = analyze(decisions, rng)
            out['served_equals_pv'] = sum(d['served'] == d['pv'] for _, d in decisions)
            out['mean_options'] = sum(len(d['options']) for _, d in decisions) / len(decisions)
            result[lineup_dir.name][phase] = out
    with output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--analyze', type=Path, help='write the per-phase analysis of --root to this path')
    args = parser.parse_args()
    if args.analyze:
        return analyze_root(args.root, args.analyze)
    cfg = json.loads(args.config.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'config.json').write_text(json.dumps({'config': cfg, 'script_sha256':
                                                       hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
    jobs = []
    for lineup_name, (lo, n) in cfg['seeds'].items():
        (args.root / lineup_name).mkdir(exist_ok=True)
        jobs += [(seed, cfg, lineup_name, str(args.root / lineup_name / f'{seed}.json')) for seed in range(lo, lo + n)]
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(run_game, jobs):
            done += 1
            print(json.dumps({'games': done, 'of': len(jobs), 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
