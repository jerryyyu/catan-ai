"""Search diagnostic: where does our planning leave wins? (Jerry 2026-11-06; evaluation, not training data.)

At sampled own-turn decisions of the PV baseline (``planner_pv2_b_fix``, seat 0) against
a declared opponent lineup, every option is evaluated by paired playouts from the TRUE
game (each playout ``r`` uses the same dice stream and reshuffled remaining deck for every
option; the same opponents as the lineup; the served bot finishes our seat's game):

- ``first``: every legal first move of the turn (builds, bank trades, development purchase,
  Year of Plenty / Road Building plays, end turn), after which the served bot plays on;
- ``seq``: the served search's candidate sequences ("end now" + top ``k`` by first move),
  each applied and followed by ending the turn (what PV search chooses among).

Per option the per-playout outcomes are stored (bits), so the analysis can split each
option's playouts in two halves: choose with one half, score with the other (unbiased
headroom), and measure how well the halves agree (label reliability). The decision also
records what the served bot and the PV bot chose.
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
from pv2_data import apply_turn  # noqa: E402

KINDS = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development', 'end_turn')


def first_moves(view, seat):
    out = []
    for a in view.actions:
        if a['type'] in KINDS or (a['type'] == 'play_development' and a.get('card') in ('year_of_plenty', 'road_building')):
            out.append(a)
    return out


def run_playout(game, seat, opponents_cls, served_seed, option, kind, stream, max_turns):
    from catan_ai.bots import make_policy
    from catan_ai.observation import Observation
    g = deepcopy(game)
    g.rng = random.Random(stream)
    g.deck = list(g.deck)
    g.rng.shuffle(g.deck)
    bots = [make_policy(name, served_seed + i) for i, name in enumerate(opponents_cls)]
    if kind == 'seq':
        if not apply_turn(g, seat, option):
            return None
    else:
        if option not in g.legal_actions(seat):
            return None
        g.apply(seat, option)
    while g.winner is None and g.turn < max_turns:
        s = g.current
        g.apply(s, bots[s].choose(Observation.for_policy(g, s, bots[s]), s))
    return int(g.winner == seat)


def run_game(args):
    seed, cfg, lineup_name, path = args
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.pv_search import candidate_sequences
    path = Path(path)
    if path.exists():
        return 'exists'
    started = time.monotonic()
    opponents = cfg['lineups'][lineup_name]
    names = ['planner_pv2_b_fix'] + opponents
    game = Game.new([f'{n}-{i}' for i, n in enumerate(names)], seed)
    bots = [make_policy(n, seed * 17 + i + 1) for i, n in enumerate(names)]
    shadow = make_policy('planner_offer_accept', seed * 17 + 1)
    playout_cls = ['planner_offer_accept'] + opponents  # served bot finishes our seat's game
    pick = random.Random(seed * 7919 + 11)
    seen, decisions = set(), []
    while game.winner is None and game.turn < cfg['max_turns'] and len(decisions) < cfg['max_decisions_per_game']:
        s = game.current
        view = Observation.for_policy(game, s, bots[s])
        first = s == 0 and game.phase == 'actions' and (game.turn, s) not in seen
        if first:
            seen.add((game.turn, s))
        if first and any(a['type'] == 'end_turn' for a in view.actions) and pick.random() < cfg['decision_rate']:
            shadow.last_offer_turn = bots[0].last_offer_turn
            served = deepcopy(shadow).choose(view, s)
            options = [('first', a) for a in first_moves(view, s)]
            options += [('seq', q) for q in candidate_sequences(shadow, view, s, cfg['candidates'])]
            rows = []
            for kind, opt in options:
                bits = []
                for r in range(cfg['playouts']):
                    w = run_playout(game, s, playout_cls, seed * 31, opt, kind,
                                    seed * 1_000_003 + game.turn * 1009 + r, cfg['max_turns'])
                    if w is None:
                        bits = None
                        break
                    bits.append(w)
                if bits is not None:
                    rows.append({'kind': kind, 'option': opt, 'bits': bits})
            action = bots[0].choose(view, s)  # the PV baseline's actual move (also advances the game)
            decisions.append({'turn': game.turn, 'points': [p.public_points for p in game.players],
                              'served': served, 'pv': action, 'options': rows})
            game.apply(s, action)
            continue
        game.apply(s, bots[s].choose(view, s))
    record = {'seed': seed, 'lineup': lineup_name, 'winner': game.winner, 'turns': game.turn,
              'seconds': time.monotonic() - started, 'decisions': decisions}
    with path.open('x') as stream:
        json.dump(record, stream)
    return 'done'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
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
