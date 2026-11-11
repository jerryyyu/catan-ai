"""Smoke run: the served policy plays classic JSettlers2 games as a third-party robot.

Starts a JSettlers server that auto-starts one ``catanai.jsettlers.CatanAiClient`` plus
built-in robots and runs robot-only games (``jsettlers.bots.botgames.*``), each with our
bot in one seat (``percent3p``; seat and first player are chosen by the server and are
recorded per game). Writes a JSON receipt with per-game outcomes, decisions, fallbacks,
bridge errors and decision latency.

Run outputs are sealed: ``--out`` (or the default stamped directory) must not exist and is
created exclusively; ``manifest.json`` (config, source and checkpoint identity, the verified
build manifest) is written with exclusive create before anything is launched, and
``receipt.json`` is published the same way, so a repeated command can never mix or overwrite
runs. The server is started in its own process group, which is killed on timeout and after
exit so no robot or bridge process outlives the run.

    uv run python scripts/jsettlers_smoke.py --config experiments/jsettlers-smoke.json

Seeded runs (opt-in; need the declared seeding patch, see tools/jsettlers/README.md): config
``jsettlers.patch`` names the patch file and its SHA-256, ``jsettlers.seed`` is the base seed
(passed as ``jsettlers.catanai.seed``; games run one at a time so game N always gets the same
name and seed) and ``jsettlers.all_smart`` makes every local robot SMART
(``jsettlers.catanai.allsmart``). The receipt then records the patch SHA-256, base seed, each
game's derived seeds, the board/first player/seats/dice the server logged, and which robots sat
with which strategy. ``--compare A B`` diffs two such receipts game by game.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import re
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsettlers_build  # noqa: E402

ROOT = jsettlers_build.ROOT


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return round(s[i], 2)


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({'type': 'corrupt_line', 'line': line[:200]})
    return rows


def board_ids(bridge_log: Path) -> dict[str, str]:
    out = {}
    for row in read_jsonl(bridge_log):
        req = row.get('request') or {}
        game = req.get('game')
        if req.get('kind') == 'decide' and game and game not in out:
            board = req['board']
            key = json.dumps({'hexes': board['hexes'], 'ports': board['ports']}, sort_keys=True)
            out[game] = hashlib.sha256(key.encode()).hexdigest()[:16]
    return out


def summarize(log_rows, bridge_log: Path, expected_games: int):
    games: dict[str, dict] = {}
    order = []

    def game(name):
        if name not in games:
            order.append(name)
            games[name] = {'game': name, 'decisions': Counter(), 'card_plays': Counter(),
                           'fallbacks': [], 'errors': [],
                           'latency_ms': [], 'legal_mismatches': [], 'finished': False}
        return games[name]

    for row in log_rows:
        kind, name = row.get('type'), row.get('game')
        if not name:
            continue
        g = game(name)
        if kind == 'game_start':
            g.update(seat=row['seat'], players=row.get('players'))
        elif kind == 'decision':
            g['decisions'][row['decision']] += 1
            action = row.get('action') or {}
            if action.get('type') == 'play_development' and row.get('cmd') == 'play_dev':
                g['card_plays'][action['card']] += 1
            g['latency_ms'].append(row['latency_ms'])
            if row.get('cmd') == 'error':
                g['errors'].append({'decision': row['decision'], 'error': row.get('error'),
                                    'turn_count': row.get('turn_count')})
            if row.get('legal_mismatches'):
                g['legal_mismatches'].append({'decision': row['decision'], 'turn_count': row.get('turn_count'),
                                              'mismatches': row['legal_mismatches']})
        elif kind == 'fallback':
            g['fallbacks'].append({'decision': row['decision'], 'reason': row.get('reason'),
                                   'turn_count': row.get('turn_count'), 'action': row.get('action')})
        elif kind == 'game_end':
            g.update(finished=True, seat=row['seat'], first_player=row['first_player'],
                     winner=row['winner'], winner_declared=row.get('winner_declared'),
                     turns=row.get('turn_count'), rounds=row.get('rounds'),
                     # The bot's client copy at game end; may lag the final server state by a message.
                     public_vp_client_copy=row.get('public_vp'))
    boards = board_ids(bridge_log)
    out = []
    for name in order:
        g = games[name]
        lat = g.pop('latency_ms')
        g['board_id'] = boards.get(name)
        g['decisions'] = dict(g['decisions'])
        g['card_plays'] = dict(g['card_plays'])
        g['decision_count'] = sum(g['decisions'].values())
        g['fallbacks_by_type'] = dict(Counter(f['decision'] for f in g['fallbacks']))
        g['bridge_errors'] = len(g['errors'])
        g['latency_ms'] = {'p50': percentile(lat, .5), 'p95': percentile(lat, .95),
                           'max': round(max(lat), 2) if lat else None, 'n': len(lat)}
        if g['finished']:
            g['our_position'] = (g['seat'] - g['first_player']) % 4
            g['won'] = g['winner'] == g['seat']
        out.append(g)
    finished = [g for g in out if g['finished']]
    all_lat = [x for row in log_rows if row.get('type') == 'decision' for x in [row['latency_ms']]]
    think = [row['think_ms'] for row in log_rows if row.get('type') == 'decision' and 'think_ms' in row]
    fallbacks = [dict(f, game=g['game']) for g in out for f in g['fallbacks']]
    wins = sum(g['won'] for g in finished)
    return {
        'games_started': len(out),
        'games_completed': len(finished),
        'truncated': max(expected_games - len(finished), 0),
        'unexpected_games': max(len(finished) - expected_games, 0),
        'our_wins': wins,
        'win_rate': round(wins / len(finished), 4) if finished else None,
        'win_rate_wilson95': wilson(wins, len(finished)),
        'seat_counts': dict(Counter(g['seat'] for g in finished)),
        'position_counts': dict(Counter(g['our_position'] for g in finished)),
        'decisions_by_type': dict(sum((Counter(g['decisions']) for g in out), Counter())),
        'fallbacks_total': len(fallbacks),
        'fallbacks_by_type': dict(Counter(f['decision'] for f in fallbacks)),
        'fallbacks': fallbacks,
        'bridge_errors': sum(g['bridge_errors'] for g in out),
        'legal_mismatch_decisions': sum(len(g['legal_mismatches']) for g in out),
        'latency_ms': {'p50': percentile(all_lat, .5), 'p95': percentile(all_lat, .95),
                       'max': round(max(all_lat), 2) if all_lat else None, 'n': len(all_lat)},
        'think_ms': {'p50': percentile(think, .5), 'p95': percentile(think, .95),
                     'max': round(max(think), 2) if think else None},
        'card_plays': dict(sum((Counter(g['card_plays']) for g in out), Counter())),
        'games': out,
    }


class RunRefused(SystemExit):
    pass


# ------------------------------------------------------------------ seeding (mirrors soc.game.CatanAiSeed)

SEED_ROLES = ('game', 'board', 'seats', 'brain0', 'brain1', 'brain2', 'brain3')
_M64 = (1 << 64) - 1


def _signed64(x: int) -> int:
    x &= _M64
    return x - (1 << 64) if x >> 63 else x


def derive_seed(base: int, game: str, role: str) -> int:
    """FNV-1a 64 over ``game + NUL + role`` from (offset basis XOR base), then SplitMix64 (Java long)."""
    h = (0xcbf29ce484222325 ^ base) & _M64
    for b in (game + '\0' + role).encode('utf-8'):
        h = ((h ^ b) * 0x100000001b3) & _M64
    z = (h + 0x9e3779b97f4a7c15) & _M64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & _M64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & _M64
    return _signed64(z ^ (z >> 31))


def game_seeds(base: int, game: str) -> dict[str, int]:
    return {role: derive_seed(base, game, role) for role in SEED_ROLES}


def policy_seed(bridge_seed: int, game: str) -> int:
    """Our policy's per-game seed (same formula as ``jsbridge.decide.Bridge.policy_for``)."""
    return (bridge_seed * 1_000_003 + zlib.crc32(game.encode())) % (2 ** 31)


def bot_strategy(name: str, all_smart: bool) -> str:
    """JSettlers' default params by name ("robot " SMART, else FAST); all-smart makes every bot SMART."""
    return 'SMART' if all_smart or name.startswith('robot ') else 'FAST'


_KV = re.compile(r'(\w+)=(\S*)')


def server_games(server_log: Path) -> dict[str, dict]:
    """Per-game facts the server printed: patch lines (catanai-*) and robot join lines."""
    games: dict[str, dict] = {}
    robots: dict[str, str] = {}
    config = None
    if not server_log.exists():
        return {}

    def g(name):
        return games.setdefault(name, {'joined': [], 'dice': []})

    for line in server_log.read_text(errors='replace').splitlines():
        if line.startswith('catanai-'):
            tag = line.split(' ', 1)[0]
            kv = dict(_KV.findall(line))
            if tag == 'catanai-config':
                config = kv
            elif tag == 'catanai-robot':
                name = line.split('name=', 1)[1].rsplit(' strategy=', 1)
                robots[name[0]] = name[1]
            elif tag == 'catanai-seed':
                g(kv['game'])['game_seed_logged'] = int(kv['game_seed'])
            elif tag == 'catanai-start':
                row = g(kv['game'])
                row['first_player'] = int(kv['first'])
                row['seats'] = line.split(' seats=', 1)[1].split(' hexes=', 1)[0].split('|')
                for key in ('hexes', 'numbers', 'ports', 'deck'):
                    m = re.search(rf' {key}=(\[[^\]]*\]|null)', line)
                    row[key] = m.group(1) if m else None
            elif tag == 'catanai-dice':
                prefix = f"{kv['roll']}:" if 'roll' in kv else ''  # roll index (patch v2 and later)
                g(kv['game'])['dice'].append(f"{prefix}{kv['turn']}:{kv['pn']}:{kv['dice']}")
        else:
            m = re.match(r'\*\*\* (.+) joined the game (\S+) at ', line)
            if m:
                g(m.group(2))['joined'].append(m.group(1))
    for row in games.values():
        row['robot_strategies'] = {n: robots.get(n) for n in row.get('seats') or row['joined'] if n}
    return {'_config': config, '_robots': robots, **games}


def decision_sequences(log_rows) -> dict[str, list]:
    seqs: dict[str, list] = {}
    for row in log_rows:
        if row.get('type') in ('decision', 'fallback') and row.get('game'):
            seqs.setdefault(row['game'], []).append(
                [row['type'], row.get('decision'), row.get('turn_count'), row.get('cmd'), row.get('action'),
                 row.get('reason')])
    return seqs


def seq_sha(seq) -> str:
    return hashlib.sha256(json.dumps(seq, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]


def seeded_game_facts(games: list[dict], log_rows, server_log: Path, jset: dict, bridge_seed: int) -> dict:
    """Attach per-game seeds, server-logged layout/dice and seated robots to each game record."""
    srv = server_games(server_log)
    seqs = decision_sequences(log_rows)
    base, all_smart = jset.get('seed'), bool(jset.get('all_smart'))
    mismatched_seeds = []
    for g in games:
        name = g['game']
        s = srv.get(name, {})
        g['policy_seed'] = policy_seed(bridge_seed, name)
        if base is not None:
            g['seeds'] = game_seeds(base, name)
            logged = s.get('game_seed_logged')
            g['seed_logged_matches'] = logged == g['seeds']['game']
            if not g['seed_logged_matches']:
                mismatched_seeds.append(name)
        for key in ('first_player', 'hexes', 'numbers', 'ports', 'deck'):
            if key in s:
                g['server_' + key] = s[key]
        if 'hexes' in s:
            g['server_board_sha'] = seq_sha([s['hexes'], s['numbers'], s['ports']])
        g['dice'] = s.get('dice', [])
        g['dice_sha'] = seq_sha(g['dice'])
        players = g.get('players') or s.get('seats') or []
        g['robots'] = [{'seat': i, 'name': n, 'strategy_expected': bot_strategy(n, all_smart),
                        'strategy_logged': (s.get('robot_strategies') or {}).get(n)}
                       for i, n in enumerate(players) if n]
        g['joined_server_log'] = s.get('joined', [])
        g['seats_server_log'] = s.get('seats')
        g['decision_seq_sha'] = seq_sha(seqs.get(name, []))
        g['decision_seq_len'] = len(seqs.get(name, []))
    return {'server_config': srv.get('_config'), 'robot_strategies_logged': srv.get('_robots'),
            'seed_mismatches': mismatched_seeds}


def jsettlers_settings(config: dict) -> dict:
    """Validate the optional patch / seed / all-smart settings; refuse inconsistent configs."""
    jset = config.get('jsettlers', {})
    seed, all_smart, patch = jset.get('seed'), bool(jset.get('all_smart')), jset.get('patch')
    if (seed is not None or all_smart) and not patch:
        raise RunRefused('jsettlers.seed / jsettlers.all_smart need the declared seeding patch (jsettlers.patch)')
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise RunRefused(f'jsettlers.seed must be an integer, got {seed!r}')
        if config['server'].get('botgames_parallel') != 1:
            raise RunRefused('a seeded run needs server.botgames_parallel = 1 (game N must get the same seed)')
    return {'seed': seed, 'all_smart': all_smart, 'patch': patch}


def server_props(jset: dict) -> list[str]:
    props = []
    if jset['seed'] is not None:
        props.append(f'-Djsettlers.catanai.seed={jset["seed"]}')
    if jset['all_smart']:
        props.append('-Djsettlers.catanai.allsmart=Y')
    return props


def write_exclusive(path: Path, data: dict) -> None:
    with open(path, 'x') as f:
        f.write(json.dumps(data, indent=2) + '\n')


def allocate_out(out: Path) -> Path:
    """Create a fresh run root; refuse (before anything else happens) if it exists."""
    out = Path(os.path.abspath(out))
    if out.exists() or out.is_symlink():
        raise RunRefused(f'refusing existing run output {out}; choose a new --out')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    return out


def bridge_identity(policy: str) -> dict:
    """What answers the robot's requests: bridge code, policy, served checkpoint, engine backend."""
    from catan_ai.engine import BACKEND
    from catan_ai.local_checkpoint import select_parameter_checkpoint
    from catan_ai.parameter_planner import load_parameters
    # The bots select the Cython-bound checkpoint when made (bots.make_policy); this identity is read first, so
    # select here too or a Cython run refuses at startup (value-arms held-out attempt 1, 2026-11-11).
    select_parameter_checkpoint()
    files = {str(p.relative_to(ROOT)): jsettlers_build.sha256(p)
             for p in [ROOT / 'scripts' / 'jsettlers_bridge.py',
                       *sorted((ROOT / 'tools' / 'jsettlers' / 'jsbridge').glob('*.py'))]}
    _, model_hash, path = load_parameters()
    return {'policy': policy, 'files': files, 'files_digest': jsettlers_build.digest(files),
            'checkpoint': str(path), 'checkpoint_model_hash': model_hash, 'engine_backend': BACKEND,
            'python': sys.executable, 'python_version': platform.python_version(),
            'parameter_model_env': os.environ.get('CATAN_PARAMETER_MODEL')}


def kill_group(pgid: int) -> None:
    """SIGTERM then SIGKILL every process in the group.

    ESRCH means the group is empty. macOS answers EPERM when the only members left are zombies
    (our own children are always signalable while running), so that also means nothing runs.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.1)


def supervise(cmd, cwd, log_path: Path, timeout: float, env=None) -> tuple[int, bool]:
    """Run ``cmd`` in its own process group; kill the whole group on timeout and after exit."""
    with open(log_path, 'x') as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        pgid = proc.pid
        try:
            rc, timed_out = proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_group(pgid)
            rc = proc.wait()
        finally:
            kill_group(pgid)  # stragglers (robot clients, bridge) never outlive the run
    return rc, timed_out


def _common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


SETUP_KEYS = ('server_board_sha', 'board_id', 'server_first_player', 'seat', 'players')
OUTCOME_KEYS = ('dice_sha', 'winner', 'turns', 'decision_seq_sha')
# Settings two receipts must share before any of their games can be paired.
COMPAT_KEYS = (('jsettlers', 'seed'), ('jsettlers', 'patch_sha256'), ('jsettlers', 'all_smart'),
               ('config', 'policy'), ('config', 'seed'), ('games_requested',))


def _dig(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _missing_facts(g: dict, decisions: list) -> list[str]:
    """Facts a game must carry before it can count as identical to anything (absent never matches)."""
    missing = [k for k in SETUP_KEYS + OUTCOME_KEYS if g.get(k) in (None, '', [])]
    if not g.get('finished'):
        missing.append('finished')
    if not g.get('dice'):
        missing.append('dice (no rolls logged)')
    if not decisions:
        missing.append('decisions (none logged)')
    elif g.get('decision_seq_sha') and seq_sha(decisions) != g['decision_seq_sha']:
        missing.append('decisions (log does not match receipt decision_seq_sha)')
    return missing


def receipt_problems(ra: dict, rb: dict) -> list[str]:
    """Receipt-level reasons the two runs cannot be paired at all."""
    problems = []
    for path in COMPAT_KEYS:
        va, vb = _dig(ra, path), _dig(rb, path)
        if va != vb:
            problems.append(f'{".".join(path)} differs: {va!r} vs {vb!r}')
    if _dig(ra, ('jsettlers', 'seed')) is None:
        problems.append('receipt is not a seeded run (jsettlers.seed missing)')
    for name, r in (('a', ra), ('b', rb)):
        names = [g.get('game') for g in r.get('games') or []]
        if not names:
            problems.append(f'receipt {name} has no games')
        if len(set(names)) != len(names) or None in names:
            problems.append(f'receipt {name} has missing or duplicate game names')
    sa = {g.get('game') for g in ra.get('games') or []}
    sb = {g.get('game') for g in rb.get('games') or []}
    if sa != sb:
        problems.append(f'game sets differ: only in a {sorted(sa - sb)[:5]}, only in b {sorted(sb - sa)[:5]}')
    return problems


def compare_receipts(path_a: Path, path_b: Path) -> dict:
    """Game-by-game comparison of two seeded receipts; fails closed on absent evidence.

    Receipts must agree on seed, patch, all-smart, policy and game count and have the same unique
    game names (``compatible``). A game with any required fact missing in either run (setup facts,
    outcome hashes, a nonempty dice log, a nonempty decision log matching its hash) is reported as
    incomplete and never counted as identical.
    """
    ra, rb = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    problems = receipt_problems(ra, rb)

    def decisions(r):
        path = (r.get('artifacts') or {}).get('decisions')
        return decision_sequences(read_jsonl(Path(path))) if path else {}

    seqs = [decisions(ra), decisions(rb)]
    gb = {g.get('game'): g for g in rb.get('games') or []}
    rows = []
    for a in ra.get('games') or []:
        name = a.get('game')
        b = gb.get(name)
        if b is None:
            rows.append({'game': name, 'incomplete': ['missing in b']})
            continue
        sa, sb = seqs[0].get(name, []), seqs[1].get(name, [])
        missing = [f'a: {m}' for m in _missing_facts(a, sa)] + [f'b: {m}' for m in _missing_facts(b, sb)]
        if missing:
            rows.append({'game': name, 'incomplete': missing})
            continue
        setup_same = {k: a[k] == b[k] for k in SETUP_KEYS}
        dice_prefix = _common_prefix(a['dice'], b['dice'])
        # Paired luck: the k-th roll's values, for every k both runs reached (turn/roller may differ).
        va, vb = ([d.rsplit(':', 1)[-1] for d in g['dice']] for g in (a, b))
        n = min(len(va), len(vb))
        dec_prefix = _common_prefix(sa, sb)
        row = {'game': name, 'setup_identical': all(setup_same.values()),
               'setup_diff': [k for k, v in setup_same.items() if not v],
               'dice': [len(a['dice']), len(b['dice'])], 'dice_common_prefix': dice_prefix,
               'dice_values_identical_for_every_k': va[:n] == vb[:n],
               'dice_values_common_prefix': _common_prefix(va, vb),
               'decisions': [len(sa), len(sb)], 'decision_common_prefix': dec_prefix,
               'winner': [a['winner'], b['winner']], 'turns': [a['turns'], b['turns']],
               'seat': a['seat'], 'first_player': a['server_first_player']}
        row['end_to_end_identical'] = row['setup_identical'] and all(a[k] == b[k] for k in OUTCOME_KEYS)
        if not row['end_to_end_identical']:
            row['first_decision_divergence'] = None if dec_prefix == max(len(sa), len(sb)) else {
                'decision_index': dec_prefix,
                'a': sa[dec_prefix] if dec_prefix < len(sa) else None,
                'b': sb[dec_prefix] if dec_prefix < len(sb) else None}
            if dice_prefix < max(len(a['dice']), len(b['dice'])):
                row['first_dice_divergence'] = {
                    'index': dice_prefix,
                    'a': a['dice'][dice_prefix] if dice_prefix < len(a['dice']) else None,
                    'b': b['dice'][dice_prefix] if dice_prefix < len(b['dice']) else None}
        rows.append(row)
    complete = [r for r in rows if 'incomplete' not in r] if not problems else []
    return {'a': str(path_a), 'b': str(path_b), 'compatible': not problems, 'problems': problems,
            'games': len(rows), 'complete_games': len(complete),
            'incomplete_games': len(rows) - len(complete),
            'end_to_end_identical': sum(r['end_to_end_identical'] for r in complete),
            'setup_identical': sum(r['setup_identical'] for r in complete),
            'setup_and_full_dice_identical': sum(r['setup_identical'] and r['dice'][0] == r['dice'][1]
                                                 == r['dice_common_prefix'] for r in complete),
            'setup_and_dice_k_identical': sum(r['setup_identical'] and r['dice_values_identical_for_every_k']
                                              for r in complete),
            'same_winner': sum(r['winner'][0] == r['winner'][1] for r in complete),
            'rows': rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--compare', nargs=2, metavar=('RECEIPT_A', 'RECEIPT_B'),
                        help='compare two seeded receipts game by game and exit')
    parser.add_argument('--out', help='new output directory (default runs/jsettlers-smoke/<utc stamp>); must not exist')
    parser.add_argument('--games', type=int, help='override the configured game count (not for receipts)')
    parser.add_argument('--finalize', metavar='RUN_DIR', help='seal receipt.json for a run killed externally after play')
    parser.add_argument('--reason', help='why the run is finalized (required with --finalize)')
    args = parser.parse_args(argv)
    if args.compare:
        result = compare_receipts(*args.compare)
        print(json.dumps(result, indent=1))
        return 0 if result['compatible'] and result['incomplete_games'] == 0 else 1
    if args.finalize:
        if not args.reason:
            parser.error('--finalize needs --reason')
        r = finalize(Path(args.finalize), args.reason)
        print(json.dumps({'games_completed': r['games_completed'], 'pass': r['pass'], **{
            k: r['finalized'][k] for k in ('unfinished_games', 'interrupted_game')}}))
        return 0
    if not args.config:
        parser.error('--config is required')
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    games = args.games or config['games']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = allocate_out(Path(args.out) if args.out else ROOT / 'runs' / 'jsettlers-smoke' / stamp)
    declared = config.get('jsettlers', {})
    if (declared.get('commit'), declared.get('tag')) != (jsettlers_build.JSETTLERS_COMMIT,
                                                         jsettlers_build.JSETTLERS_TAG):
        raise RunRefused(f'config pins JSettlers {declared}, build verifies '
                         f'{jsettlers_build.JSETTLERS_TAG} {jsettlers_build.JSETTLERS_COMMIT}')
    jset = jsettlers_settings(config)
    decision_log, bridge_log = out / 'decisions.jsonl', out / 'bridge.jsonl'
    server_log, bridge_err = out / 'server.log', out / 'bridge.stderr'

    # Private class path inside this run: no other run or build can change what this JVM loads.
    build = jsettlers_build.build_manifest(out / 'build', patch=jset['patch'])
    build_dir = Path(build['build_dir'])
    write_exclusive(out / 'build-manifest.json', build)
    manifest = {
        'name': config['name'],
        'created_utc': stamp,
        'config_path': str(config_path),
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'config': config,
        'games_requested': games,
        'source_revision': git('rev-parse', 'HEAD'),
        'source_dirty': bool(git('status', '--porcelain', '--', 'tools/jsettlers', 'scripts', 'src', 'checkpoints')),
        'build': {'manifest_sha256': jsettlers_build.sha256(out / 'build-manifest.json'),
                  'jsettlers': {k: v for k, v in build['jsettlers'].items() if k != 'files'},
                  'jdk_version': build['jdk']['version'].splitlines()[0],
                  'bridge_java_sources_digest': build['bridge_java']['sources_digest'],
                  'bridge_java_classes_digest': build['bridge_java']['classes_digest']},
        'bridge': bridge_identity(config['policy']),
        'jsettlers': {'patch_sha256': build['jsettlers']['patch_sha256'], 'seed': jset['seed'],
                      'all_smart': jset['all_smart'], 'build_dir': str(build_dir)},
    }
    write_exclusive(out / 'manifest.json', manifest)

    srv = config['server']
    bridge_cmd = ' '.join([sys.executable, str(ROOT / 'scripts' / 'jsettlers_bridge.py'),
                           '--policy', config['policy'], '--seed', str(config['seed']),
                           '--log', str(bridge_log)])
    cmd = [jsettlers_build.java(), '-cp', jsettlers_build.classpath(build_dir=build_dir),
           f'-Dcatanai.bridge.cmd={bridge_cmd}', f'-Dcatanai.bridge.cwd={ROOT}',
           f'-Dcatanai.bridge.stderr={bridge_err}', f'-Dcatanai.decision.log={decision_log}',
           f'-Dcatanai.bridge.timeout_ms={int(config.get("bridge_timeout_ms", 30000))}',
           'soc.server.SOCServer',
           f'-Djsettlers.startrobots={srv["builtin_robots"]}',
           '-Djsettlers.bots.start3p=1,catanai.jsettlers.CatanAiClient',
           f'-Djsettlers.bots.percent3p={srv["percent3p"]}',
           f'-Djsettlers.bots.botgames.total={games}',
           f'-Djsettlers.bots.botgames.parallel={srv["botgames_parallel"]}',
           f'-Djsettlers.bots.botgames.wait_sec={srv["botgames_wait_sec"]}',
           '-Djsettlers.bots.botgames.shutdown=Y',
           f'-Djsettlers.bots.fast_pause_percent={srv["fast_pause_percent"]}',
           f'-Djsettlers.bots.timeout.turn={srv["timeout_turn_sec"]}',
           *server_props(jset),
           str(srv['port']), str(srv['max_connections'])]
    jsettlers_build.verify_loaded(build)  # nothing on the class path changed since verification
    started = time.time()
    rc, timed_out = supervise(cmd, build_dir, server_log,
                              config['timeout_sec_per_game'] * games, env=dict(os.environ))
    elapsed = round(time.time() - started, 1)
    receipt = build_receipt(out, manifest, config, games, jset, rc, timed_out, elapsed)
    write_exclusive(out / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('games_completed', 'our_wins', 'bridge_errors', 'fallbacks_total',
                                              'fallbacks_by_type', 'latency_ms', 'elapsed_sec', 'pass')}))
    print(f'receipt: {out / "receipt.json"}')
    return 0 if receipt['pass'] else 1


def build_receipt(out: Path, manifest: dict, config: dict, games: int, jset: dict, rc, timed_out, elapsed) -> dict:
    """The receipt from a run directory's sealed logs (shared by a normal run and ``--finalize``)."""
    decision_log, bridge_log = out / 'decisions.jsonl', out / 'bridge.jsonl'
    server_log, bridge_err = out / 'server.log', out / 'bridge.stderr'
    log_rows = read_jsonl(decision_log)
    summary = summarize(log_rows, bridge_log, games)
    seeding = seeded_game_facts(summary['games'], log_rows, server_log, jset, config['seed'])
    retired = [r for r in read_jsonl(decision_log) if r.get('type') == 'bridge_retired']
    receipt = {
        **{k: manifest[k] for k in ('name', 'created_utc', 'config_path', 'config_sha256', 'config',
                                    'games_requested', 'source_revision', 'source_dirty', 'build', 'bridge',
                                    'jsettlers')},
        'seeding': seeding,
        'manifest_sha256': jsettlers_build.sha256(out / 'manifest.json'),
        'server_rc': rc,
        'timed_out': timed_out,
        'elapsed_sec': elapsed,
        'bridge_retired': retired,
        'artifacts': {k: str(v) for k, v in {'decisions': decision_log, 'bridge': bridge_log,
                                             'server_log': server_log, 'bridge_stderr': bridge_err}.items()},
        **summary,
    }
    bar = config.get('pass_bar', {})
    receipt['pass'] = (summary['games_completed'] >= min(bar.get('completed_games', games), games)
                       and summary['games_completed'] <= games
                       and summary['bridge_errors'] <= bar.get('bridge_errors', 0)
                       and not retired and not timed_out
                       and not seeding['seed_mismatches'])
    return receipt


def last_started_game(server_log: Path) -> str | None:
    """The last game the server started (``catanai-start`` lines; botgames run one at a time)."""
    last = None
    with server_log.open(errors='replace') as stream:
        for line in stream:
            if line.startswith('catanai-start '):
                for part in line.split():
                    if part.startswith('game='):
                        last = part[len('game='):]
    return last


def finalize(out: Path, reason: str) -> dict:
    """Seal ``receipt.json`` for a run whose runner process was killed externally after its games were played.

    The receipt is rebuilt from the run's own sealed manifest and logs with ``build_receipt`` (the code a normal
    run uses). It records the interruption: no server rc, no elapsed time, the reason, this script's revision and
    hash, and the one game the server had started last (``interrupted_game``) if it never finished. The strict
    ``pass`` is computed exactly as for a normal run, so an unfinished game keeps it false; whether such a game may
    be excluded is the analyzer's predeclared decision (``jsettlers_paired.py --allow-interrupted``).
    """
    out = Path(out)
    if (out / 'receipt.json').exists():
        raise RunRefused(f'{out} already has a receipt')
    for name in ('manifest.json', 'build-manifest.json', 'decisions.jsonl', 'bridge.jsonl', 'server.log'):
        if not (out / name).exists():
            raise RunRefused(f'{out}/{name} missing; nothing to finalize')
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['build']['manifest_sha256'] != jsettlers_build.sha256(out / 'build-manifest.json'):
        raise RunRefused('build manifest changed after the run sealed it')
    config, games = manifest['config'], manifest['games_requested']
    receipt = build_receipt(out, manifest, config, games, jsettlers_settings(config), None, False, None)
    unfinished = [g['game'] for g in receipt['games'] if not g.get('finished') or g.get('won') is None]
    started_last = last_started_game(out / 'server.log')
    receipt['finalized'] = {
        'reason': reason, 'finalized_utc': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        'finalize_source_revision': git('rev-parse', 'HEAD'),
        'finalize_script_sha256': jsettlers_build.sha256(Path(__file__)),
        'manifest_source_revision': manifest['source_revision'],
        'last_started_game': started_last, 'unfinished_games': unfinished,
        'interrupted_game': started_last if unfinished == [started_last] else None}
    write_exclusive(out / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    raise SystemExit(main())
