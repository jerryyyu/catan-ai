"""Headroom probe for turn-exact search (board #1 row "Turn-exact search"; Jerry 2026-11-09).

The served ``planner_pv2_tiles`` scores only five end-of-turn candidates: end now plus the first
moves of the served planner's four best build sequences (a 128-node beam search ranked by the
economic leaf). Turn-exact search would instead enumerate every afterstate reachable this turn with
builds, bank/port trades and a development purchase (a purchase ends the sequence, as in the served
search), deduplicated by the resulting own position, and score all of them with the same tiles value.

This probe plays seeded self-play games of ``planner_pv2_tiles`` (all four seats) and, at every
decision where the PV search runs, records: the number of sequences expanded and unique afterstates,
whether the enumeration hit its node cap, the PV pick's value and the exhaustive best's value (both
the tiles model's P(win) of the afterstate, purchases averaged over the deck prior), and whether the
exhaustive best is an afterstate PV never scored. It is a diagnostic: the values are the model's own
estimates, so a gap says where exact search would act differently, not that it would win more. That
needs playouts or a screen.

    uv run --with numpy python scripts/exact_turn_probe.py --seeds 2990000 20 --out runs/exact-probe-v1.json
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.encoding import encode  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from catan_ai.pv_search import DECK_PRIOR, REPLACEABLE, afterstate_variants, candidate_sequences  # noqa: E402

EXPAND = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development')
POLICY = 'planner_pv2_tiles'


def _key(game, seat, bought):
    p = game.players[seat]
    return (tuple(sorted((str(r), n) for r, n in p.resources.items())), tuple(sorted(p.settlements)),
            tuple(sorted(p.cities)), tuple(sorted(tuple(sorted(e)) for e in p.roads)), bought)


def enumerate_turn(planner, view, seat, max_nodes):
    """Unique own-turn afterstates reachable from ``view``: {key: (seq, world, bought)}."""
    from copy import deepcopy
    root = planner._world(view)
    out = {_key(root, seat, False): ([], root, False)}
    stack, nodes, capped = [(root, [])], 0, False
    while stack:
        game, seq = stack.pop()
        for action in game.legal_actions(seat):
            if action['type'] not in EXPAND:
                continue
            if nodes >= max_nodes:
                capped = True
                stack = []
                break
            nodes += 1
            child = deepcopy(game, {id(game.board): game.board})
            bought = action['type'] == 'buy_development'
            if bought:
                if not child.players[seat].can_afford('development'):
                    continue
                child.players[seat].pay('development')
            else:
                child.apply(seat, action)
            key = _key(child, seat, bought)
            if key in out:
                continue
            out[key] = (seq + [action], child, bought)
            if not bought and child.winner is None:
                stack.append((child, seq + [action]))
    return out, nodes, capped


def value_of(planner, world, seat, bought, turn):
    """The tiles model's P(win) for an afterstate (purchases: mean over the deck prior)."""
    from copy import deepcopy
    if world.winner == seat:
        return 1.0
    if not bought:
        rows, weights = [encode(Observation.from_game(world, seat, include_events=False), world)], [1.0]
    else:
        total = sum(n for _, n in DECK_PRIOR)
        rows, weights = [], []
        for card, n in DECK_PRIOR:
            variant = deepcopy(world, {id(world.board): world.board})
            variant.players[seat].development.append((card, turn))
            rows.append(encode(Observation.from_game(variant, seat, include_events=False), variant))
            weights.append(n / total)
    logits = planner._logits(rows)
    return sum(w / (1 + math.exp(-z)) for w, z in zip(weights, logits))


def pv_value(planner, view, seat, seq):
    variants = afterstate_variants(planner, view, seat, seq)
    if variants is None:
        return None
    if variants == 'won':
        return 1.0
    logits = planner._logits([x for _, x in variants])
    return sum(w / (1 + math.exp(-z)) for (w, _), z in zip(variants, logits))


def probe_decision(planner, view, seat, max_nodes):
    t0 = time.perf_counter()
    cands = candidate_sequences(planner, view, seat, planner.K)
    pv = [(seq, pv_value(planner, view, seat, seq)) for seq in cands]
    pv = [(s, v) for s, v in pv if v is not None]
    pv_best_seq, pv_best = max(pv, key=lambda t: t[1])
    t1 = time.perf_counter()
    states, nodes, capped = enumerate_turn(planner, view, seat, max_nodes)
    t2 = time.perf_counter()
    best_v, best_seq = -1.0, None
    for seq, world, bought in states.values():
        v = value_of(planner, world, seat, bought, view.turn)
        if v > best_v:
            best_v, best_seq = v, seq
    t3 = time.perf_counter()
    return {'turn': view.turn, 'seat': seat, 'pv_candidates': len(pv), 'nodes': nodes, 'unique': len(states),
            'capped': capped, 'pv_best': round(pv_best, 5), 'exact_best': round(best_v, 5),
            'gap': round(best_v - pv_best, 5), 'same_first_move': (best_seq or [None])[:1] == (pv_best_seq or [None])[:1],
            'exact_len': len(best_seq or []), 'pv_seconds': round(t1 - t0, 4), 'enum_seconds': round(t2 - t1, 4),
            'score_seconds': round(t3 - t2, 4)}


def run(seed, games, max_nodes, max_turns=1000):
    rows = []
    for g in range(games):
        s = seed + g
        game = Game.new([f'{POLICY}-{i}' for i in range(4)], s)
        bots = [make_policy(POLICY, s * 17 + i + 1) for i in range(4)]
        while game.winner is None and game.turn < max_turns:
            seat = game.current
            view = Observation.for_policy(game, seat, bots[seat])
            bot = bots[seat]
            if view.phase == 'actions' and any(a['type'] == 'end_turn' for a in view.actions):
                before = bot.last_offer_turn
                from catan_ai.unified_planner import AcceptanceOfferPlanner
                served = AcceptanceOfferPlanner.choose(bot, view, seat)
                bot.last_offer_turn = before
                if served['type'] in REPLACEABLE and served['type'] not in ('offer_trade', 'play_development'):
                    rows.append({'seed': s, **probe_decision(bot, view, seat, max_nodes)})
            game.apply(seat, bot.choose(view, seat))
    return rows


def summarize(rows):
    gaps = [r['gap'] for r in rows]
    return {'decisions': len(rows), 'capped': sum(r['capped'] for r in rows),
            'unique_p50': statistics.median(r['unique'] for r in rows), 'unique_max': max(r['unique'] for r in rows),
            'exact_better_share': round(sum(g > 1e-6 for g in gaps) / len(rows), 4),
            'different_first_move_share': round(sum(not r['same_first_move'] for r in rows) / len(rows), 4),
            'gap_mean': round(statistics.mean(gaps), 5),
            'gap_p90': round(sorted(gaps)[int(.9 * (len(gaps) - 1))], 5), 'gap_max': round(max(gaps), 5),
            'enum_plus_score_seconds_p50': statistics.median(r['enum_seconds'] + r['score_seconds'] for r in rows),
            'enum_plus_score_seconds_p95': sorted(r['enum_seconds'] + r['score_seconds'] for r in rows)[int(.95 * (len(rows) - 1))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs=2, type=int, required=True, metavar=('START', 'COUNT'))
    parser.add_argument('--max-nodes', type=int, default=4000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    t0 = time.time()
    rows = run(args.seeds[0], args.seeds[1], args.max_nodes)
    result = {'policy': POLICY, 'seeds': args.seeds, 'max_nodes': args.max_nodes, 'summary': summarize(rows),
              'elapsed_seconds': round(time.time() - t0, 1), 'rows': rows}
    with args.out.open('x') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result['summary'], indent=1))


if __name__ == '__main__':
    main()
