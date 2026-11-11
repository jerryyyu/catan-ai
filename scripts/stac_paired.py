"""Paired STAC ladder table: two policies on the same seeded StacSettlers games (experiments/stac-ladder.json).

Both receipts come from ``scripts/stac_match.py`` runs with the same STAC seed, patch, agents, blocks,
trading mode and game count, differing only in our policy. Seeding pins the environment (board, deck,
first player, seats, dice by roll index), so games are paired by name, and a pair counts only when
both runs finished it with identical setup evidence (``stac_match.SETUP_KEYS``). STAC's MCTS play is
not reproducible, so pairing removes board/seat/dice luck but not opponent-play variance.

Reported per opponent: each policy's win share with a Wilson 95% interval, the retirement rule (share
>= .50 with Wilson lower bound > .40), and the paired difference won(A) - won(B) with a two-sided
paired-normal 95% interval.

Health (Codex HOLD @42aad77): the declared coverage is an input (``--expect OPPONENT=GAMES``, from the
predeclaration), never inferred from the receipts. Each receipt must carry, as typed values, exactly that
many unique game records per opponent, every one finished with an outcome; ``games_requested`` and
``total.games_completed`` equal to the declared total; ``total.truncated``, ``total.bridge_errors`` and
offer timeouts 0; and empty ``observability_violations``, ``seed_mismatches`` and
``stac_results_disagree`` lists (present, not just falsy). Any of these failing makes the table
unhealthy. Among otherwise complete declared games, pairs whose setup evidence is missing or differs
are excluded and counted; more than 5% excluded fails the table.

    python scripts/stac_paired.py A/receipt.json B/receipt.json \
        --expect pomcp_ts_cr=256 --expect stac_negotiator=512 --out OUT.json
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


def restrict(r: dict, expect: dict) -> dict:
    """The receipt restricted to the blocks whose opponent is declared in ``expect`` (owner decision 2026-11-11:
    STAC negotiator dropped from the promotion bar). Totals, runs and games are rebuilt from the kept blocks' own
    evidence; a receipt whose blocks are all declared is returned unchanged."""
    blocks = (r.get('config') or {}).get('blocks') or []
    if not all(isinstance(b, dict) and 'opponent' in b and 'name' in b for b in blocks):
        return r  # no per-opponent block declarations: nothing to restrict (health() judges it as is)
    keep = [b for b in blocks if b.get('opponent') in expect]
    if len(keep) == len(blocks):
        return r
    names = {b['name'] for b in keep}
    per = {n: v for n, v in (r.get('blocks') or {}).items() if n in names}
    if set(per) != names:
        return {**r, 'total': None}  # missing per-block evidence: health() refuses it
    total = {}
    for key in ('games', 'games_completed', 'truncated', 'bridge_errors'):
        vals = [v.get(key) for v in per.values()]
        total[key] = sum(vals) if all(_is_int(x) for x in vals) else None
    for key in ('observability_violations', 'seed_mismatches', 'stac_results_disagree'):
        vals = [v.get(key) for v in per.values()]
        total[key] = [x for v in vals for x in v] if all(isinstance(v, list) for v in vals) else None
    chats = [v.get('chat_trades') for v in per.values()]
    total['chat_trades'] = ({'offer_timeout': sum(c.get('offer_timeout', 0) for c in chats)}
                            if all(isinstance(c, dict) for c in chats) else None)
    return {**r, 'config': {**r['config'], 'blocks': keep}, 'games_requested': sum(b['games'] for b in keep),
            'blocks': per, 'total': total, 'runs': [x for x in r.get('runs') or [] if x.get('block') in names],
            'games': [g for g in r.get('games') or [] if g.get('block') in names],
            'restricted_from': sorted(b['name'] for b in blocks if b['name'] not in names)}


def _dig(r, path):
    for k in path:
        r = (r or {}).get(k)
    return r


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def health(r: dict, tag: str, expect: dict[str, int]) -> list[str]:
    """Coverage and health evidence one receipt must carry before its games can be used."""
    t = r.get('total')
    if not isinstance(t, dict):
        return [f'{tag}: no total block']
    problems = []
    declared = sum(expect.values())
    if r.get('games_requested') != declared:
        problems.append(f'{tag}: games_requested {r.get("games_requested")!r} != declared {declared}')
    for key in ('games_completed', 'truncated', 'bridge_errors'):
        if not _is_int(t.get(key)):
            problems.append(f'{tag}: missing {key} evidence')
    if t.get('games_completed') != declared:
        problems.append(f'{tag}: games_completed {t.get("games_completed")!r} != declared {declared}')
    for key in ('truncated', 'bridge_errors'):
        if _is_int(t.get(key)) and t[key] != 0:
            problems.append(f'{tag}: {key} {t[key]}')
    chat = t.get('chat_trades')
    if not isinstance(chat, dict):
        problems.append(f'{tag}: missing chat_trades evidence')
    elif chat.get('offer_timeout', 0) != 0:
        problems.append(f'{tag}: {chat["offer_timeout"]} offer timeouts')
    for key in ('observability_violations', 'seed_mismatches', 'stac_results_disagree'):
        if not isinstance(t.get(key), list):
            problems.append(f'{tag}: missing {key} evidence')
        elif t[key]:
            problems.append(f'{tag}: {key} {t[key][:3]}')
    runs = r.get('runs')
    if not isinstance(runs, list) or not runs or any(x.get('timed_out') is not False for x in runs):
        problems.append(f'{tag}: block run evidence missing or a block timed out')
    games = r.get('games')
    if not isinstance(games, list):
        return problems + [f'{tag}: no game records']
    names = [g.get('game') for g in games]
    if None in names or len(set(names)) != len(names):
        problems.append(f'{tag}: missing or duplicate game names')
    counts = {}
    for g in games:
        counts[g.get('opponent')] = counts.get(g.get('opponent'), 0) + 1
    if counts != expect:
        problems.append(f'{tag}: game records per opponent {counts} != declared {expect}')
    unfinished = [g.get('game') for g in games if g.get('finished') is not True or not isinstance(g.get('won'), bool)]
    if unfinished:
        problems.append(f'{tag}: {len(unfinished)} games without a finished outcome, e.g. {unfinished[:3]}')
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


def compare(ra: dict, rb: dict, expect: dict[str, int], max_excluded: float = 0.05) -> dict:
    if not expect or any(not _is_int(n) or n <= 0 for n in expect.values()):
        raise ValueError('declared coverage (--expect OPPONENT=GAMES) is required')
    ra, rb = restrict(ra, expect), restrict(rb, expect)
    problems = [f'{".".join(p)} differs' for p in COMPAT if _dig(ra, p) != _dig(rb, p)]
    if _dig(ra, ('config', 'policy')) == _dig(rb, ('config', 'policy')):
        problems.append('both receipts use the same policy')
    problems += health(ra, 'a', expect) + health(rb, 'b', expect)
    ga = {g['game']: g for g in ra.get('games') or []}
    gb = {g['game']: g for g in rb.get('games') or []}
    by_opp, excluded = {}, []
    for name in sorted(set(ga) | set(gb)):
        a, b = ga.get(name), gb.get(name)
        if a is None or b is None:
            problems.append(f'game {name} missing in one run')
            continue
        if not (a.get('finished') is True and b.get('finished') is True) or not isinstance(a.get('won'), bool) \
                or not isinstance(b.get('won'), bool):
            continue  # already a health problem above
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
    return {'expect': expect, 'restricted_from': {'a': ra.get('restricted_from'), 'b': rb.get('restricted_from')},
            'a_policy': _dig(ra, ('config', 'policy')), 'b_policy': _dig(rb, ('config', 'policy')),
            'compatible': not problems, 'problems': problems, 'games_total': total,
            'games_excluded': len(excluded), 'excluded': excluded[:50], 'tables': tables,
            'healthy': not problems and not too_many,
            **({'fail_reason': f'more than {max_excluded:.0%} of pairs excluded'} if too_many else {})}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect', action='append', required=True, metavar='OPPONENT=GAMES',
                        help='declared games per opponent (experiments/stac-ladder.json), repeat per opponent')
    args = parser.parse_args(argv)
    expect = {}
    for item in args.expect:
        opp, _, n = item.partition('=')
        expect[opp] = int(n)
    result = compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()), expect)
    result['a_receipt'], result['b_receipt'] = str(args.a), str(args.b)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'excluded'}, indent=1))
    return 0 if result['healthy'] else 1


if __name__ == '__main__':
    sys.exit(main())
