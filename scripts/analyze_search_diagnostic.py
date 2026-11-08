"""Analysis for ``search_diagnostic.py`` (predeclared in experiments/search-diagnostic.json).

Per decision, each option's 64 playout outcomes are split into halves A (even index) and
B (odd index). Metrics, per lineup, with a game-clustered bootstrap 95% CI:

- reliability: Pearson correlation, over all non-served options, between the half-A and
  half-B estimates of (option rate minus served-move rate);
- headroom of a candidate set: choose the set's best option on half A and score it on
  half B minus the served move on half B (and B->A, averaged). Estimand: the gain over
  the served bot (one step, with the served bot playing on) of a *finite-sample
  selector* that picks the option with the best rate on 32 noisy playouts, scored on 32
  independent playouts. The independent scoring removes the winner's-curse optimism of
  in-sample maxima, but the selector is noisy: this is neither the gain of a perfect
  oracle choosing within the set nor an upper bound on what search over the set could
  achieve (a better evaluator, e.g. more playouts, picks better). With the observed
  split-half reliability (.23-.30), much of the true spread between options is masked,
  so a small estimate does not establish that candidate-set search cannot reach the
  +5 pp screen bar. Sets: ``seq`` (served "end now" + top sequences, what PV picks
  from), ``first`` (every legal first move), and ``all`` (both);
- PV realized: the PV move's full-sample rate minus the served move's (both mapped to
  their first-move options);
- best-move types: the move type of the half-A best option in ``first``.

The served and PV moves are mapped to the ``first`` option with the same first action;
a served/PV trade offer or end of turn maps to "end turn" (offers are not simulated —
a stated limitation).

Coverage (Codex HOLD @c32a17a, P2): the root is read through
``search_diagnostic.declared_records``, which refuses it unless every declared seed of
every lineup is present, nothing undeclared is, and every record matches the root's
``manifest.json`` identity (``--legacy``: a pre-manifest root, coverage checked only,
reported ``legacy_unbound``). ``analyze`` itself is unchanged.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rate(bits):
    return sum(bits) / len(bits) if bits else 0.


def halves(bits):
    return bits[0::2], bits[1::2]


def first_key(action):
    if action['type'] in ('offer_trade', 'end_turn'):
        return json.dumps({'type': 'end_turn'}, sort_keys=True)
    return json.dumps(action, sort_keys=True)


def analyze(decisions, rng, resamples=1000):
    rows = []
    rel_a, rel_b = [], []
    types = Counter()
    for g, d in decisions:
        first = {json.dumps(o['option'], sort_keys=True): o for o in d['options'] if o['kind'] == 'first'}
        served = first.get(first_key(d['served']))
        pv = first.get(first_key(d['pv']))
        if served is None:
            continue
        sA, sB = halves(served['bits'])
        sets = {'seq': [o for o in d['options'] if o['kind'] == 'seq'],
                'first': list(first.values()), 'all': d['options']}
        out = {'game': g}
        for name, opts in sets.items():
            if not opts:
                out[name] = 0.
                continue
            pa = max(opts, key=lambda o: rate(halves(o['bits'])[0]))
            pb = max(opts, key=lambda o: rate(halves(o['bits'])[1]))
            out[name] = .5 * ((rate(halves(pa['bits'])[1]) - rate(sB)) + (rate(halves(pb['bits'])[0]) - rate(sA)))
        out['pv'] = (rate(pv['bits']) - rate(served['bits'])) if pv is not None else 0.
        best = max(first.values(), key=lambda o: rate(halves(o['bits'])[0]))
        types[best['option']['type']] += 1
        for o in d['options']:
            if o is served:
                continue
            a, b = halves(o['bits'])
            rel_a.append(rate(a) - rate(sA))
            rel_b.append(rate(b) - rate(sB))
        rows.append(out)
    by_game = {}
    for r in rows:
        by_game.setdefault(r['game'], []).append(r)
    ids = sorted(by_game)

    def mean(rs, k):
        return sum(r[k] for r in rs) / len(rs)
    summary = {}
    for k in ('seq', 'first', 'all', 'pv'):
        draws = sorted(mean([r for gid in (rng.choice(ids) for _ in ids) for r in by_game[gid]], k) for _ in range(resamples))
        summary[k] = {'mean': mean(rows, k), 'ci95': [draws[int(.025 * resamples)], draws[int(.975 * resamples)]]}
    try:
        reliability = st.correlation(rel_a, rel_b) if len(rel_a) > 2 else None
    except st.StatisticsError:  # constant inputs (tiny samples)
        reliability = None
    return {'decisions': len(rows), 'games': len(ids), 'headroom_vs_served': summary,
            'split_half_reliability': reliability, 'best_first_move_types': dict(types)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy', action='store_true', help='analyze a pre-manifest root (config.json only)')
    args = parser.parse_args()
    from search_diagnostic import declared_records
    records, identity = declared_records(args.root, args.legacy)
    rng = random.Random(29)
    result = {}
    for lineup, recs in records.items():
        decisions = []
        for rec in recs:
            decisions += [(rec['seed'], d) for d in rec['decisions']]
        result[lineup] = analyze(decisions, rng)
    result['_identity'] = identity
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
