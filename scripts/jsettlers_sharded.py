"""Sharded seeded JSettlers run: K servers play disjoint slices of one declared table concurrently.

A seeded ``scripts/jsettlers_smoke.py`` run plays its games one at a time (``botgames_parallel`` 1) so game
N always gets the name ``~botsOnly~N`` and the seeds derived from (``jsettlers.seed``, name). One server
therefore cannot be split, but K servers can: shard k of K runs its own smoke config with

    jsettlers.seed  = base * 1000 + k        (k < 1000, so the map from (base, k) is injective)
    games           = N // K, plus one for each of the first N % K shards (the slices sum to N)
    server.port     = server.port + k * 100  (PORT_STRIDE; every declared JSettlers port is 88xx)
    out             = OUT/shard-k            (sealed by jsettlers_smoke exactly as an unsharded run)

Everything else in the config (policy, policy seed, patch, all_smart, server settings, pass bar) is
copied, and a ``shard`` block records the parent. The parent config declares ``"shards": K`` (absent or 1
is an ordinary unsharded run, which this runner refuses). The runner refuses an existing OUT, K < 2,
K > N, K >= 1000, an unseeded config, a seed outside the Java long range, a shard port outside
1025..65535, a shard port declared (directly or as a shard) by another experiment config, and a shard
port that is busy right now.

It writes ``OUT/manifest.json`` (exclusive) before launching, runs the K smoke subprocesses concurrently,
then merges the K sealed shard receipts into ``OUT/receipt.json`` (exclusive). In the merged receipt game
``~botsOnly~N`` of shard k is named ``s{k}:~botsOnly~N``; counts are summed; ``jsettlers.seed`` is the
declared base, with ``jsettlers.shards``, ``shard_seed_function`` and ``shard_seeds`` naming the slices;
``shards`` lists each shard's config/receipt SHA-256, seed, port and pass. ``pass`` holds only when every
shard passed (rc 0), the shards agree on source/bridge/patch/build identity, and the summed totals meet
the declared ``pass_bar``. A missing or unreadable shard receipt refuses the merge (no merged receipt).

Candidate and control must declare the same K (the seed function is fixed here), so merged names pair
1:1; ``scripts/jsettlers_paired.py`` refuses a sharded receipt against an unsharded one or a different
K / shard seed list.

    uv run python scripts/jsettlers_sharded.py --config experiments/X.json --plan
    uv run python scripts/jsettlers_sharded.py --config experiments/X.json --out runs/X
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'scripts' / 'jsettlers_smoke.py'
PORT_STRIDE = 100
SEED_STRIDE = 1000
MAX_SHARDS = SEED_STRIDE - 1
SHARD_SEED_FUNCTION = f'base * {SEED_STRIDE} + k'
_JAVA_LONG = 1 << 63

# Totals that add across shards.
SUM_KEYS = ('games_started', 'games_completed', 'truncated', 'unexpected_games', 'our_wins',
            'fallbacks_total', 'bridge_errors', 'legal_mismatch_decisions')
COUNTER_KEYS = ('seat_counts', 'position_counts', 'decisions_by_type', 'fallbacks_by_type', 'card_plays')
# Identity every shard must share for the merged table to be one design.
SHARED_IDENTITY = (('source_revision', ('source_revision',)),
                   ('patch_sha256', ('jsettlers', 'patch_sha256')),
                   ('all_smart', ('jsettlers', 'all_smart')),
                   ('bridge files', ('bridge', 'files_digest')),
                   ('checkpoint', ('bridge', 'checkpoint_model_hash')),
                   ('engine backend', ('bridge', 'engine_backend')),
                   ('jsettlers build', ('build', 'jsettlers')))


class ShardRefused(SystemExit):
    pass


def shard_name(k: int, game: str) -> str:
    return f's{k}:{game}'


def shard_seed(base: int, k: int) -> int:
    seed = base * SEED_STRIDE + k
    if not -_JAVA_LONG <= seed < _JAVA_LONG:
        raise ShardRefused(f'shard seed {seed} for base {base} leaves the Java long range')
    return seed


def split_games(total: int, shards: int) -> list[int]:
    """Even split; the first ``total % shards`` shards get one extra game."""
    q, r = divmod(total, shards)
    return [q + (1 if k < r else 0) for k in range(shards)]


def shard_ports(config: dict) -> list[int]:
    k = config.get('shards', 1)
    k = k if isinstance(k, int) and not isinstance(k, bool) and k >= 1 else 1
    return [config['server']['port'] + i * PORT_STRIDE for i in range(k)]


def declared_ports(experiments: Path, exclude: Path | None = None) -> dict[int, str]:
    """Ports other experiment configs declare (their base port and, if sharded, every shard port)."""
    out: dict[int, str] = {}
    for path in sorted(experiments.glob('*.json')):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        try:
            ports = shard_ports(json.loads(path.read_text()))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        for p in ports:
            out.setdefault(p, path.name)
    return out


def shard_configs(config: dict, config_sha256: str, other_ports: dict[int, str] | None = None) -> list[dict]:
    """Derive the K shard configs from a parent config declaring ``shards: K``; refuse unsafe designs."""
    k_total = config.get('shards', 1)
    if not isinstance(k_total, int) or isinstance(k_total, bool) or k_total < 2:
        raise ShardRefused(f'shards must be an integer >= 2 for a sharded run, got {k_total!r}')
    if k_total > MAX_SHARDS:
        raise ShardRefused(f'shards {k_total} > {MAX_SHARDS} would break the shard seed map ({SHARD_SEED_FUNCTION})')
    games = config.get('games')
    if not isinstance(games, int) or isinstance(games, bool) or k_total > games:
        raise ShardRefused(f'cannot split games={games!r} into {k_total} nonempty shards')
    jset = config.get('jsettlers') or {}
    base = jset.get('seed')
    if not isinstance(base, int) or isinstance(base, bool) or not jset.get('patch'):
        raise ShardRefused('a sharded run must be seeded (integer jsettlers.seed and the declared jsettlers.patch)')
    if (config.get('server') or {}).get('botgames_parallel') != 1:
        raise ShardRefused('a sharded run needs server.botgames_parallel = 1 in every shard')
    ports = shard_ports(config)
    for p in ports:
        if not 1024 < p <= 65535:
            raise ShardRefused(f'shard port {p} outside 1025..65535')
        if other_ports and p in other_ports:
            raise ShardRefused(f'shard port {p} collides with a port declared by {other_ports[p]}')
    out = []
    for k, (n, port) in enumerate(zip(split_games(games, k_total), ports)):
        cfg = copy.deepcopy(config)
        cfg.pop('shards')
        cfg['name'] = f'{config["name"]}-shard{k}'
        cfg['games'] = n
        cfg['jsettlers']['seed'] = shard_seed(base, k)
        cfg['server']['port'] = port
        cfg['shard'] = {'index': k, 'of': k_total, 'parent_name': config['name'],
                        'parent_config_sha256': config_sha256, 'parent_games': games, 'base_seed': base,
                        'seed_function': SHARD_SEED_FUNCTION, 'port_stride': PORT_STRIDE}
        out.append(cfg)
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96):
    """Same interval as jsettlers_smoke.wilson (kept local so this module imports nothing heavy)."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [round(c - h, 4), round(c + h, 4)]


def _counter_sum(rows) -> dict:
    total = Counter()
    for row in rows:
        total.update({str(k): v for k, v in (row or {}).items()})
    return dict(total)


def _dig(d, path):
    for key in path:
        d = d.get(key) if isinstance(d, dict) else None
    return d


def merge_receipts(config: dict, config_sha256: str, shard_receipts: list[dict],
                   shard_files: list[dict] | None = None) -> dict:
    """Merge K sealed shard receipts (in shard order) into one table receipt; fail closed on any gap."""
    expected = shard_configs(config, config_sha256)
    k_total = len(expected)
    if len(shard_receipts) != k_total or any(not isinstance(r, dict) for r in shard_receipts):
        raise ShardRefused(f'{len(shard_receipts)} shard receipts for shards={k_total}; no merged receipt')
    base = config['jsettlers']['seed']
    n = config['games']
    problems = []
    for k, (r, cfg) in enumerate(zip(shard_receipts, expected)):
        if r.get('config') != cfg:
            problems.append(f'shard {k}: receipt config differs from the derived shard {k} config')
        if _dig(r, ('jsettlers', 'seed')) != cfg['jsettlers']['seed']:
            problems.append(f'shard {k}: seed {_dig(r, ("jsettlers", "seed"))!r} != {cfg["jsettlers"]["seed"]}')
        if r.get('games_requested') != cfg['games']:
            problems.append(f'shard {k}: games_requested {r.get("games_requested")!r} != {cfg["games"]}')
    for label, path in SHARED_IDENTITY:
        if len({json.dumps(_dig(r, path), sort_keys=True) for r in shard_receipts}) != 1:
            problems.append(f'shards disagree on {label}')

    games, fallbacks, retired, mismatches, shards = [], [], [], [], []
    files = shard_files or [{} for _ in range(k_total)]
    for k, r in enumerate(shard_receipts):
        games += [dict(g, game=shard_name(k, g['game']), shard=k, shard_game=g['game']) for g in r.get('games') or []]
        fallbacks += [dict(f, game=shard_name(k, f['game'])) for f in r.get('fallbacks') or []]
        retired += [dict(x, shard=k) for x in r.get('bridge_retired') or []]
        mismatches += [shard_name(k, g) for g in _dig(r, ('seeding', 'seed_mismatches')) or []]
        shards.append({'index': k, 'seed': _dig(r, ('jsettlers', 'seed')), 'port': _dig(r, ('config', 'server', 'port')),
                       'games_requested': r.get('games_requested'), 'games_completed': r.get('games_completed'),
                       'config_sha256': r.get('config_sha256'), 'manifest_sha256': r.get('manifest_sha256'),
                       'receipt': files[k].get('receipt'), 'receipt_sha256': files[k].get('receipt_sha256'),
                       'pass': r.get('pass'), 'server_rc': r.get('server_rc'), 'timed_out': r.get('timed_out'),
                       'elapsed_sec': r.get('elapsed_sec'), 'latency_ms': r.get('latency_ms'),
                       'think_ms': r.get('think_ms'), 'server_config': _dig(r, ('seeding', 'server_config'))})
    totals = {key: sum(int(r.get(key) or 0) for r in shard_receipts) for key in SUM_KEYS}
    for k, r in enumerate(shard_receipts):
        finished = [g for g in r.get('games') or [] if g.get('finished')]
        if (r.get('games_completed'), r.get('our_wins')) != (len(finished), sum(bool(g.get('won')) for g in finished)):
            problems.append(f'shard {k}: games_completed/our_wins disagree with its game records')
    lat = [r.get('latency_ms') or {} for r in shard_receipts]
    first = shard_receipts[0]
    timed_out = any(bool(r.get('timed_out')) for r in shard_receipts)
    receipt = {
        'name': config['name'], 'config_sha256': config_sha256, 'config': config, 'games_requested': n,
        'source_revision': first.get('source_revision'),
        'source_dirty': any(bool(r.get('source_dirty')) for r in shard_receipts),
        'bridge': first.get('bridge'), 'build': first.get('build'),
        'jsettlers': {'patch_sha256': _dig(first, ('jsettlers', 'patch_sha256')), 'seed': base,
                      'all_smart': _dig(first, ('jsettlers', 'all_smart')), 'shards': k_total,
                      'shard_seed_function': SHARD_SEED_FUNCTION, 'port_stride': PORT_STRIDE,
                      'shard_seeds': [c['jsettlers']['seed'] for c in expected],
                      'shard_games': [c['games'] for c in expected]},
        'seeding': {'seed_mismatches': mismatches},
        'shards': shards,
        'merge_problems': problems,
        'server_rc': [r.get('server_rc') for r in shard_receipts],
        'timed_out': timed_out,
        'shard_elapsed_sec': [r.get('elapsed_sec') for r in shard_receipts],
        'bridge_retired': retired,
        **totals,
        'win_rate': round(totals['our_wins'] / totals['games_completed'], 4) if totals['games_completed'] else None,
        'win_rate_wilson95': (wilson(totals['our_wins'], totals['games_completed'])
                              if totals['our_wins'] <= totals['games_completed'] else None),
        **{key: _counter_sum(r.get(key) for r in shard_receipts) for key in COUNTER_KEYS},
        'fallbacks': fallbacks,
        # Percentiles do not add across shards; each shard's p50/p95 is in ``shards``.
        'latency_ms': {'max': max((x['max'] for x in lat if x.get('max') is not None), default=None),
                       'n': sum(int(x.get('n') or 0) for x in lat), 'p50': None, 'p95': None},
        'games': games,
    }
    bar = config.get('pass_bar', {})
    names = [g['game'] for g in games]
    receipt['pass'] = bool(
        not problems
        and all(r.get('pass') is True for r in shard_receipts)
        and len(names) == len(set(names))
        and min(bar.get('completed_games', n), n) <= totals['games_completed'] <= n
        and totals['bridge_errors'] <= bar.get('bridge_errors', 0)
        and not retired and not timed_out and not mismatches)
    return receipt


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', port))
        except OSError:
            return True
    return False


def write_exclusive(path: Path, data: dict) -> None:
    with open(path, 'x') as f:
        f.write(json.dumps(data, indent=2) + '\n')


def git(*args) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run(config_path: Path, out: Path) -> int:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    out = Path(os.path.abspath(out))
    if out.exists() or out.is_symlink():
        raise ShardRefused(f'refusing existing run output {out}; choose a new --out')
    configs = shard_configs(config, config_sha256, declared_ports(ROOT / 'experiments', exclude=config_path))
    busy = [c['server']['port'] for c in configs if port_busy(c['server']['port'])]
    if busy:
        raise ShardRefused(f'shard ports busy now: {busy}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    paths = []
    for k, cfg in enumerate(configs):
        p = out / f'shard-{k}.config.json'
        write_exclusive(p, cfg)
        paths.append(p)
    manifest = {'name': config['name'], 'created_utc': stamp, 'config_path': str(config_path),
                'config_sha256': config_sha256, 'config': config, 'games_requested': config['games'],
                'source_revision': git('rev-parse', 'HEAD'),
                'source_dirty': bool(git('status', '--porcelain', '--', 'tools/jsettlers', 'scripts', 'src',
                                         'checkpoints')),
                'shard_seed_function': SHARD_SEED_FUNCTION, 'port_stride': PORT_STRIDE,
                'shards': [{'index': k, 'config': str(p), 'config_sha256': sha256_file(p),
                            'seed': c['jsettlers']['seed'], 'port': c['server']['port'], 'games': c['games'],
                            'out': str(out / f'shard-{k}')} for k, (p, c) in enumerate(zip(paths, configs))]}
    write_exclusive(out / 'manifest.json', manifest)

    started = time.time()
    procs = []
    try:
        for k, p in enumerate(paths):
            log = open(out / f'shard-{k}.log', 'x')
            proc = subprocess.Popen([sys.executable, str(SMOKE), '--config', str(p), '--out', str(out / f'shard-{k}')],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
            procs.append((proc, log))
        rcs = [proc.wait() for proc, _ in procs]
    except BaseException:
        # SIGINT lets each smoke runner's own cleanup kill its server process group.
        for proc, _ in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc, _ in procs:
            proc.wait()
        raise
    finally:
        for _, log in procs:
            log.close()
    elapsed = round(time.time() - started, 1)

    receipts, files = [], []
    for k in range(len(configs)):
        rp = out / f'shard-{k}' / 'receipt.json'
        if not rp.is_file():
            raise ShardRefused(f'shard {k} wrote no receipt (rc {rcs[k]}); see {out / f"shard-{k}.log"}; '
                               'no merged receipt')
        try:
            receipts.append(json.loads(rp.read_text()))
        except ValueError as exc:
            raise ShardRefused(f'shard {k} receipt unreadable ({exc}); no merged receipt') from exc
        files.append({'receipt': str(rp), 'receipt_sha256': sha256_file(rp)})
    receipt = merge_receipts(config, config_sha256, receipts, files)
    receipt.update(created_utc=stamp, config_path=str(config_path), elapsed_sec=elapsed, shard_rc=rcs,
                   manifest_sha256=sha256_file(out / 'manifest.json'))
    receipt['pass'] = bool(receipt['pass'] and all(rc == 0 for rc in rcs))
    write_exclusive(out / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('games_completed', 'our_wins', 'bridge_errors', 'fallbacks_total',
                                              'merge_problems', 'elapsed_sec', 'pass')}))
    print(f'receipt: {out / "receipt.json"}')
    return 0 if receipt['pass'] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='experiment config declaring "shards": K >= 2')
    parser.add_argument('--out', type=Path, help='new run root; must not exist')
    parser.add_argument('--plan', action='store_true', help='print the derived shard seeds/games/ports and exit')
    args = parser.parse_args(argv)
    if args.plan:
        raw = args.config.read_bytes()
        cfgs = shard_configs(json.loads(raw), hashlib.sha256(raw).hexdigest(),
                             declared_ports(ROOT / 'experiments', exclude=args.config))
        print(json.dumps([{'name': c['name'], 'games': c['games'], 'seed': c['jsettlers']['seed'],
                           'port': c['server']['port']} for c in cfgs], indent=1))
        return 0
    if args.out is None:
        parser.error('--out is required unless --plan')
    return run(args.config, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
