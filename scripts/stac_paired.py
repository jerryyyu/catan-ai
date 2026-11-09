"""Paired STAC ladder table: two policies on the same seeded StacSettlers games (experiments/stac-ladder.json).

Both receipts come from ``scripts/stac_match.py`` runs with the same STAC seed, patch, agents, blocks,
trading mode and game count, differing only in our policy. Seeding pins the environment (board, deck,
first player, seats, dice by roll index), so games are paired by name, and a pair counts only when
both runs finished it with identical setup evidence (``stac_match.SETUP_KEYS``). STAC's MCTS play is
not reproducible, so pairing removes board/seat/dice luck but not opponent-play variance.

Reported per opponent and overall: each policy's win share with a Wilson 95% interval, the retirement
rule (share >= .50 with Wilson lower bound > .40), and the paired difference won(A) - won(B) with a
two-sided paired-normal 95% interval. Health per receipt: games completed == requested, 0 bridge
errors, 0 offer timeouts, observability as declared; more than 5% of pairs excluded fails the table.

    python scripts/stac_paired.py A/receipt.json B/receipt.json --out OUT.json
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stac_match as sm  # noqa: E402

COMPAT = (('stac', 'seed'), ('stac', 'patch_sha256'), ('stac', 'player_trading'), ('config', 'blocks'),
          ('config', 'agents'), ('config', 'seed'), ('games_requested',))


def _dig(r, path):
    for k in path:
        r = (r or {}).get(k)
    return r


def health(r: dict, tag: str) -> list[str]:
    t = r.get('total') or {}
    problems = []
    if t.get('games_completed') != r.get('games_requested'):
        problems.append(f'{tag}: {t.get("games_completed")} of {r.get("games_requested")} games completed')
    if t.get('bridge_errors') != 0:
        problems.append(f'{tag}: bridge_errors {t.get("bridge_errors")}')
    if (t.get('chat_trades') or {}).get('offer_timeout', 0):
        problems.append(f'{tag}: {t["chat_trades"]["offer_timeout"]} offer timeouts')
    for key in ('observability_violations', 'seed_mismatches', 'stac_results_disagree'):
        if t.get(key):
            problems.append(f'{tag}: {key} {t[key][:3]}')
    if any(x.get('timed_out') for x in r.get('runs') or []):
        problems.append(f'{tag}: a block timed out')
    return problems


def share(wins: int, n: int) -> dict:
    lo, hi = sm.wilson(wins, n) if n else (None, None)
    return {'wins': wins, 'games': n, 'share': round(wins / n, 4) if n else None, 'wilson95': [lo, hi],
            'retired': bool(n and wins / n >= .5 and lo > .4)}


def paired(rows: list[tuple[int, int]]) -> dict:
    d = [a - b for a, b in rows]
    out = {'pairs': len(d)}
    if len(d) >= 2:
        mean, sd = statistics.mean(d), statistics.stdev(d)
        se = sd / math.sqrt(len(d))
        out.update(mean_diff=round(mean, 4), sd=round(sd, 4), ci95=[round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)])
    return out


def compare(ra: dict, rb: dict, max_excluded: float = 0.05) -> dict:
    problems = [f'{".".join(p)} differs' for p in COMPAT if _dig(ra, p) != _dig(rb, p)]
    if _dig(ra, ('config', 'policy')) == _dig(rb, ('config', 'policy')):
        problems.append('both receipts use the same policy')
    problems += health(ra, 'a') + health(rb, 'b')
    ga = {g['game']: g for g in ra.get('games') or []}
    gb = {g['game']: g for g in rb.get('games') or []}
    by_opp, excluded = {}, []
    for name in sorted(set(ga) | set(gb)):
        a, b = ga.get(name), gb.get(name)
        if a is None or b is None:
            excluded.append({'game': name, 'reason': 'missing in one run'})
            continue
        if not (a.get('finished') and b.get('finished')) or a.get('won') is None or b.get('won') is None:
            excluded.append({'game': name, 'reason': 'unfinished'})
            continue
        if any(a.get(k) in (None, '', []) or a.get(k) != b.get(k) for k in sm.SETUP_KEYS):
            excluded.append({'game': name, 'reason': 'setup evidence missing or different'})
            continue
        by_opp.setdefault(a['opponent'], []).append((int(bool(a['won'])), int(bool(b['won']))))
    tables = {}
    for opp, rows in sorted(by_opp.items()):
        tables[opp] = {'a': share(sum(x for x, _ in rows), len(rows)), 'b': share(sum(y for _, y in rows), len(rows)),
                       'paired_a_minus_b': paired(rows)}
    total = len(set(ga) | set(gb))
    too_many = total == 0 or len(excluded) > max_excluded * total
    return {'a_policy': _dig(ra, ('config', 'policy')), 'b_policy': _dig(rb, ('config', 'policy')),
            'compatible': not problems, 'problems': problems, 'games_total': total,
            'games_excluded': len(excluded), 'excluded': excluded[:50], 'tables': tables,
            'healthy': not problems and not too_many,
            **({'fail_reason': f'more than {max_excluded:.0%} of pairs excluded'} if too_many else {})}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
    result['a_receipt'], result['b_receipt'] = str(args.a), str(args.b)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'excluded'}, indent=1))
    return 0 if result['healthy'] else 1


if __name__ == '__main__':
    sys.exit(main())
