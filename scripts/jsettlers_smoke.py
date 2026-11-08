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
import time
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
    from catan_ai.parameter_planner import load_parameters
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', help='new output directory (default runs/jsettlers-smoke/<utc stamp>); must not exist')
    parser.add_argument('--games', type=int, help='override the configured game count (not for receipts)')
    args = parser.parse_args(argv)
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
    decision_log, bridge_log = out / 'decisions.jsonl', out / 'bridge.jsonl'
    server_log, bridge_err = out / 'server.log', out / 'bridge.stderr'

    build = jsettlers_build.build_manifest()
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
    }
    write_exclusive(out / 'manifest.json', manifest)

    srv = config['server']
    bridge_cmd = ' '.join([sys.executable, str(ROOT / 'scripts' / 'jsettlers_bridge.py'),
                           '--policy', config['policy'], '--seed', str(config['seed']),
                           '--log', str(bridge_log)])
    cmd = [jsettlers_build.java(), '-cp', jsettlers_build.classpath(),
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
           str(srv['port']), str(srv['max_connections'])]
    jsettlers_build.verify_loaded(build)  # nothing on the class path changed since verification
    started = time.time()
    rc, timed_out = supervise(cmd, jsettlers_build.BUILD, server_log,
                              config['timeout_sec_per_game'] * games, env=dict(os.environ))
    elapsed = round(time.time() - started, 1)
    summary = summarize(read_jsonl(decision_log), bridge_log, games)
    retired = [r for r in read_jsonl(decision_log) if r.get('type') == 'bridge_retired']
    receipt = {
        **{k: manifest[k] for k in ('name', 'created_utc', 'config_path', 'config_sha256', 'config',
                                    'games_requested', 'source_revision', 'source_dirty', 'build', 'bridge')},
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
                       and not retired and not timed_out)
    write_exclusive(out / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('games_completed', 'our_wins', 'bridge_errors', 'fallbacks_total',
                                              'fallbacks_by_type', 'latency_ms', 'elapsed_sec', 'pass')}))
    print(f'receipt: {out / "receipt.json"}')
    return 0 if receipt['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
