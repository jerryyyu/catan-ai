"""Paired JSettlers table: candidate vs control on the same seeded games (experiments/tiles-heldout.json).

Both receipts come from scripts/jsettlers_smoke.py runs of the seeded (patch v2) JSettlers server with the
same base seed, patch, all_smart setting and game count, differing only in our policy. Games are paired by
name; a pair counts only when both runs carry complete setup evidence and the setups are identical (board,
first player, our seat, seated robots, deck), so both policies faced the same luck. The statistic is the
per-game difference won(candidate) − won(control); the one-sided paired-normal lower bound uses the alpha
given on the command line.

Merged receipts from scripts/jsettlers_sharded.py pair the same way (by their ``s{k}:~botsOnly~N`` names), but
only with another sharded receipt of the same K, seed function, shard seeds and shard game counts, whose merge
recorded no problems; a sharded receipt never pairs with an unsharded one.

Usage:
    python scripts/jsettlers_paired.py CANDIDATE/receipt.json CONTROL/receipt.json --alpha 0.0125 --expect-games 512 --out OUT.json
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsettlers_smoke as smoke  # noqa: E402

# The two runs differ in our policy by design; every other compatibility key must match.
POLICY_FREE_PROBLEMS = ('config.policy differs',)
SETUP_PAIR_KEYS = smoke.SETUP_KEYS + ('server_deck',)


def normal_quantile(p: float) -> float:
    return statistics.NormalDist().inv_cdf(p)


def setup_identity(g: dict) -> tuple:
    return tuple(json.dumps(g.get(k), sort_keys=True) for k in SETUP_PAIR_KEYS)


SHARD_KEYS = ('shards', 'shard_seed_function', 'shard_seeds', 'shard_games')


def shard_problems(cand: dict, ctrl: dict) -> list[str]:
    """Sharded (merged) receipts pair only with sharded receipts of the identical shard design.

    Two unsharded receipts carry none of these keys, so this adds nothing for them.
    """
    ja, jb = cand.get('jsettlers') or {}, ctrl.get('jsettlers') or {}
    sharded = ['shards' in ja, 'shards' in jb]
    if not any(sharded):
        return []
    if not all(sharded):
        return [f'sharded receipt cannot pair with an unsharded one (candidate shards={ja.get("shards")!r}, '
                f'control shards={jb.get("shards")!r})']
    problems = [f'jsettlers.{k} differs: {ja.get(k)!r} vs {jb.get(k)!r}' for k in SHARD_KEYS if ja.get(k) != jb.get(k)]
    for name, r, j in (('candidate', cand, ja), ('control', ctrl, jb)):
        k = j.get('shards')
        if not isinstance(k, int) or k < 2 or len(r.get('shards') or []) != k:
            problems.append(f'{name}: sharded receipt lists {len(r.get("shards") or [])} shards for shards={k!r}')
        if r.get('merge_problems') != []:
            problems.append(f'{name}: shard merge problems {r.get("merge_problems")!r}')
    return problems


def health_problems(r: dict, name: str, expect_games: int | None) -> list[str]:
    """Completed-coverage evidence a receipt must carry before it can support a confirmatory verdict."""
    problems = []
    games = r.get('games') or []
    requested = r.get('games_requested')
    if expect_games is not None and requested != expect_games:
        problems.append(f'{name}: games_requested {requested!r} != declared {expect_games}')
    for key in ('games_completed', 'truncated', 'bridge_errors'):
        if not isinstance(r.get(key), int):
            problems.append(f'{name}: missing {key} evidence')
    if isinstance(r.get('games_completed'), int) and r.get('games_completed') != requested:
        problems.append(f'{name}: games_completed {r.get("games_completed")} != games_requested {requested}')
    if r.get('truncated') not in (0, None) or (isinstance(r.get('truncated'), int) and r['truncated'] != 0):
        problems.append(f'{name}: truncated {r.get("truncated")}')
    if isinstance(r.get('bridge_errors'), int) and r['bridge_errors'] != 0:
        problems.append(f'{name}: bridge_errors {r["bridge_errors"]}')
    if len(games) != requested:
        problems.append(f'{name}: {len(games)} game records != games_requested {requested}')
    unfinished = [g.get('game') for g in games if not g.get('finished') or g.get('won') is None]
    if unfinished:
        problems.append(f'{name}: {len(unfinished)} unfinished games, e.g. {unfinished[:3]}')
    return problems


def pair(cand: dict, ctrl: dict, alpha: float, max_excluded: float = 0.05, expect_games: int | None = None,
         point_bar: float | None = None) -> dict:
    problems = [p for p in smoke.receipt_problems(cand, ctrl)
                if not any(p.startswith(prefix) for prefix in POLICY_FREE_PROBLEMS)]
    problems += shard_problems(cand, ctrl)
    problems += health_problems(cand, 'candidate', expect_games) + health_problems(ctrl, 'control', expect_games)
    if (cand.get('config') or {}).get('policy') == (ctrl.get('config') or {}).get('policy'):
        problems.append('candidate and control use the same policy')
    games_c = {g['game']: g for g in cand.get('games') or [] if g.get('game')}
    games_b = {g['game']: g for g in ctrl.get('games') or [] if g.get('game')}
    rows, excluded = [], []
    for name in sorted(set(games_c) | set(games_b)):
        a, b = games_c.get(name), games_b.get(name)
        if a is None or b is None:
            excluded.append({'game': name, 'reason': 'missing in one run'})
            continue
        missing = [k for k in SETUP_PAIR_KEYS if a.get(k) in (None, '', []) or b.get(k) in (None, '', [])]
        if missing or not a.get('finished') or not b.get('finished') or a.get('won') is None or b.get('won') is None:
            excluded.append({'game': name, 'reason': f'incomplete: {missing or "finish/outcome"}'})
            continue
        if setup_identity(a) != setup_identity(b):
            excluded.append({'game': name, 'reason': 'setup differs between runs'})
            continue
        rows.append({'game': name, 'diff': int(bool(a['won'])) - int(bool(b['won']))})
    n_total = len(set(games_c) | set(games_b))
    diffs = [r['diff'] for r in rows]
    result = {'compatible': not problems, 'problems': problems, 'games_total': n_total, 'games_paired': len(rows),
              'games_excluded': len(excluded), 'excluded': excluded[:50], 'alpha': alpha,
              'candidate_policy': (cand.get('config') or {}).get('policy'),
              'control_policy': (ctrl.get('config') or {}).get('policy'),
              'candidate_wins_paired': sum(int(bool(games_c[r['game']]['won'])) for r in rows),
              'control_wins_paired': sum(int(bool(games_b[r['game']]['won'])) for r in rows)}
    if len(diffs) >= 2:
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        se = sd / math.sqrt(len(diffs))
        result.update(mean_diff=mean, sd=sd, se=se, lcb=mean - normal_quantile(1 - alpha) * se,
                      ci95=[mean - 1.96 * se, mean + 1.96 * se])
    too_many_excluded = n_total == 0 or len(excluded) > max_excluded * n_total
    result['expect_games'] = expect_games
    result['pass'] = bool(result['compatible'] and not too_many_excluded and len(diffs) >= 2
                          and result['mean_diff'] >= 0 and result['lcb'] > 0)
    if too_many_excluded:
        result['fail_reason'] = f'more than {max_excluded:.0%} of games excluded'
    if point_bar is not None:
        # A DEV screen's verdict: the paired point lift clears its bar, with the same health checks.
        result['point_bar'] = point_bar
        result['dev_pass'] = bool(result['compatible'] and not too_many_excluded and len(diffs) >= 2
                                  and result['mean_diff'] >= point_bar)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('control', type=Path)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--point-bar', type=float, default=None,
                        help='DEV verdict: dev_pass = healthy and paired point lift >= this bar (the LCB pass is still reported)')
    parser.add_argument('--expect-games', type=int, required=True,
                        help='the declared game count of each run (512 for experiments/tiles-heldout.json)')
    args = parser.parse_args()
    cand = json.loads(args.candidate.read_text())
    ctrl = json.loads(args.control.read_text())
    result = pair(cand, ctrl, args.alpha, expect_games=args.expect_games, point_bar=args.point_bar)
    result['candidate_receipt'] = str(args.candidate)
    result['control_receipt'] = str(args.control)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'excluded'}, indent=1))
    verdict = result['dev_pass'] if args.point_bar is not None else result['pass']
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
