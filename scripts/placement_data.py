"""Placement training data (Jerry 2026-11-07: 'Yes and allow' — one targeted data run
for a fast placement model; config experiments/placement-data.json).

Games: each seat's style is drawn from STYLES by the game seed. At EVERY
``setup_settlement`` decision (8 per game) the candidates are the served bot's top
``top_k`` vertices plus ``random_k`` uniformly drawn other legal vertices; each is
applied to a copy of the true game and played to the end ``playouts`` times by the
game's real lineup (deck order and dice sampled per playout; common random numbers).
Setup has no hidden hands, so the afterstate features are exact public information.

Per candidate row: encoder-v2 features of the afterstate from the chooser's view
(``x``), wins/playouts, whether it is the served choice, the real choice; per
decision: the opponents' true styles (seat-relative) and the public style posterior
(``style_posterior``) at that moment. The real game continues with the seat's own bot.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import time


def run_game(job):
    seed, cfg, path = job
    from catan_ai.bots import make_policy
    from catan_ai.encoding import encode
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.placement_search import STYLES, style_posterior, DECK
    path = Path(path)
    if path.exists():
        return 'exists'
    started = time.monotonic()
    rng = random.Random(seed * 7 + 3)
    lineup = [rng.choice(STYLES) for _ in range(4)]
    game = Game.new([f'p{i}' for i in range(4)], seed)
    bots = [make_policy(n, seed * 17 + i + 1) for i, n in enumerate(lineup)]
    served = make_policy('planner_placement_search', seed)  # for the served ranking only
    decisions = []
    while game.winner is None and game.turn < cfg['max_turns']:
        s = game.current
        view = Observation.for_policy(game, s, bots[s])
        if game.phase == 'setup_settlement':
            hist = Observation.for_policy(game, s, served)
            ranked = served._served_ranking(hist, s)
            top = ranked[:cfg['top_k']]
            rest = ranked[cfg['top_k']:]
            pick = random.Random(seed * 1009 + game.setup_index)
            cands = top + pick.sample(rest, min(cfg['random_k'], len(rest)))
            rows = []
            for c in cands:
                after = deepcopy(game)
                after.apply(s, c)
                x = encode(Observation.from_game(after, s, include_events=False), after)
                wins = 0
                for r in range(cfg['playouts']):
                    g = deepcopy(game)
                    g.rng = random.Random(seed * 1_000_003 + game.setup_index * 10_007 + r)
                    g.deck = list(DECK)
                    g.rng.shuffle(g.deck)
                    pbots = [make_policy(n, seed * 31 + r * 4 + i) for i, n in enumerate(lineup)]
                    g.apply(s, c)
                    while g.winner is None and g.turn < cfg['max_turns']:
                        t = g.current
                        g.apply(t, pbots[t].choose(Observation.for_policy(g, t, pbots[t]), t))
                    wins += g.winner == s
                rows.append({'vertex': c['vertex'], 'x': [round(v, 5) for v in x], 'wins': wins,
                             'playouts': cfg['playouts'], 'served': c == ranked[0],
                             'served_rank': ranked.index(c)})
            post = style_posterior(hist, s)
            action = bots[s].choose(view, s)
            decisions.append({'seat': s, 'setup_index': game.setup_index, 'style': lineup[s],
                              'opponent_styles': [lineup[(s + k) % 4] for k in (1, 2, 3)],
                              'posterior': [list(post[(s + k) % 4][1]) for k in (1, 2, 3)],
                              'chosen_vertex': action['vertex'], 'candidates': rows})
            game.apply(s, action)
            continue
        game.apply(s, bots[s].choose(view, s))
    record = {'seed': seed, 'lineup': lineup, 'winner': game.winner, 'turns': game.turn,
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
    (args.root / 'games').mkdir(parents=True, exist_ok=True)
    meta = args.root / 'config.json'
    ident = {'config': cfg, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if meta.exists():
        if json.loads(meta.read_text()) != ident:
            raise ValueError('resume identity differs')
    else:
        meta.write_text(json.dumps(ident, indent=1))
    jobs = []
    for split, (lo, n) in cfg['split'].items():
        jobs += [(seed, cfg, str(args.root / 'games' / f'{seed}.json')) for seed in range(lo, lo + n)]
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(run_game, jobs):
            done += 1
            print(json.dumps({'games': done, 'of': len(jobs), 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
