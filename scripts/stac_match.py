"""Seeded StacSettlers matches: the served policy (one seat) against STAC's own agents.

Runs STAC's own simulation harness (``soc.robot.stac.simulation.Simulation``, in-process server and
robot clients, no network) from a private, verified class path (``scripts/stac_build.py``: pinned
StacSettlers commit + the declared Catan AI match patch). Each config block is one Simulation JVM:
``games`` games of our bot (``class:catanai.stac.CatanAiStacFactory``) against three copies of one
STAC agent:

* ``pomcp_ts_cr``: Dobre's POMCP with human action-type seeding and conditioned typed rollouts
  (AIIDE 2018 "POMCP-TS-CR"; 10k iterations, 4 threads, C=4, PUCT, seeding temperature 7, at most
  3 offers), with player trading off (``NO_TRADES``) and a declared per-decision time cap
  (``MCTS_TIME_LIMIT_MS``; every search is logged with its iterations and whether the cap cut it);
* ``jsettlers``: STAC's bundled JSettlers 1.1.16 robot (default FAST parameters).

Match settings are fixed and checked: FullyObservable and ObservableVP off (the patched server logs
both per game and the receipt fails closed if either is on), bank and port trades on. Player trading
is one of two declared modes (``stac.player_trading``):

* ``off`` (default): chat negotiation off, ``Trades=false`` for the JSettlers brains, ``NO_TRADES``
  for MCTS, the bridge disables our offers, and the patched server refuses any player-to-player
  exchange and logs the attempt; the receipt fails if any player trade executed.
* ``chat``: STAC's chat negotiation on (``ChatNegotiation=true``, ``Trades=true``, no ``NO_TRADES``,
  no server block). STAC's agents negotiate in their chat trade format, and our brain speaks it
  (``catanai.stac.CatanAiStacBrain``): it answers offers with the policy's accept/reject and sends the
  policy's offers. Every chat trade event is logged and counted in the receipt.

 ``stac.seed`` seeds board, deck, first player, seats, dice by roll index,
steals, forced discards and robot tie-breaks (see ``soc.game.CatanAiSeed`` in the patch).

Outputs are sealed like the JSettlers runner: ``--out`` must not exist; ``manifest.json`` (config
bytes, source revision, verified build manifest, bridge identity) is written before launch and
``receipt.json`` after, both with exclusive create. ``--compare A B`` diffs two seeded receipts
game by game and fails closed on missing evidence.

    uv run python scripts/stac_match.py --config experiments/stac-smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsettlers_smoke as js  # noqa: E402  (shared run hygiene helpers; not modified here)
import stac_build  # noqa: E402

ROOT = stac_build.ROOT
OUR_NAME = 'CatanAi'
OUR_FACTORY = 'class:catanai.stac.CatanAiStacFactory'
AGENT_NAMES = {'pomcp_ts_cr': 'POMCPTSCR', 'jsettlers': 'JSettlers'}
AGENT_FACTORIES = {'pomcp_ts_cr': 'mcts', 'jsettlers': 'jsettlers'}

wilson, percentile, read_jsonl = js.wilson, js.percentile, js.read_jsonl
write_exclusive, allocate_out, supervise = js.write_exclusive, js.allocate_out, js.supervise
derive_seed, seq_sha, RunRefused = js.derive_seed, js.seq_sha, js.RunRefused


# ------------------------------------------------------------------ config

def check_config(config: dict) -> dict:
    """Validate match settings; refuse anything that would weaken the boundary or the seeding."""
    stac = config.get('stac') or {}
    if (stac.get('repo'), stac.get('commit')) != (stac_build.STAC_REPO, stac_build.STAC_COMMIT):
        raise RunRefused(f'config pins StacSettlers {stac.get("repo")} {stac.get("commit")}, build verifies '
                         f'{stac_build.STAC_REPO} {stac_build.STAC_COMMIT}')
    if not stac.get('patch'):
        raise RunRefused('stac.patch (file + sha256) is required')
    for key in ('fully_observable', 'observable_vp'):
        if stac.get(key) is not False:
            raise RunRefused(f'stac.{key} must be false for a bridge match')
    mode = trading_mode(stac)
    if mode == 'off':
        if stac.get('chat_negotiation') is not False:
            raise RunRefused('stac.chat_negotiation must be false when player_trading is off')
        if stac.get('no_player_trades') is not True:
            raise RunRefused('stac.no_player_trades must be true (player trading is off for the match)')
    else:
        if stac.get('chat_negotiation') is not True:
            raise RunRefused('stac.chat_negotiation must be true when player_trading is chat')
        if stac.get('no_player_trades') is not False:
            raise RunRefused('stac.no_player_trades must be false when player_trading is chat')
    seed = stac.get('seed')
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise RunRefused(f'stac.seed must be an integer, got {seed!r}')
    agents = config.get('agents') or {}
    names = set()
    for block in config.get('blocks') or []:
        if block.get('opponent') not in AGENT_NAMES or block['opponent'] not in agents:
            raise RunRefused(f'block {block.get("name")} has unknown opponent {block.get("opponent")!r}')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', block.get('name', '')) or block['name'] in names:
            raise RunRefused(f'bad or duplicate block name {block.get("name")!r}')
        names.add(block['name'])
        if not isinstance(block.get('games'), int) or block['games'] < 1:
            raise RunRefused(f'block {block["name"]} needs a positive game count')
    if not names:
        raise RunRefused('config has no blocks')
    mcts = agents.get('pomcp_ts_cr')
    if mcts is not None:
        flags = mcts.get('flags') or []
        if mode == 'off' and 'NO_TRADES' not in flags:
            raise RunRefused('pomcp_ts_cr must carry NO_TRADES (player trading is off for the match)')
        if mode == 'chat' and 'NO_TRADES' in flags:
            raise RunRefused('pomcp_ts_cr must not carry NO_TRADES when player_trading is chat')
        if not isinstance(mcts.get('time_cap_ms'), int) or mcts['time_cap_ms'] <= 0:
            raise RunRefused('pomcp_ts_cr needs a declared per-decision time_cap_ms')
        if any(f.startswith('MCTS_TIME_LIMIT_MS') for f in flags):
            raise RunRefused('declare the cap in time_cap_ms, not in flags')
    return stac


def trading_mode(stac: dict) -> str:
    mode = stac.get('player_trading', 'off')
    if mode not in ('off', 'chat'):
        raise RunRefused(f'stac.player_trading must be "off" or "chat", got {mode!r}')
    return mode


def agent_line(kind: str, agents: dict) -> str:
    spec = agents[kind]
    flags = list(spec.get('flags') or [])
    if kind == 'pomcp_ts_cr':
        flags.append(f'MCTS_TIME_LIMIT_MS:{spec["time_cap_ms"]}')
    line = f'Agent=3,{AGENT_NAMES[kind]},{AGENT_FACTORIES[kind]}'
    return line + (',' + '|'.join(flags) if flags else '')


def simulation_config(block: dict, agents: dict, mode: str = 'off') -> str:
    """STAC Simulation config text for one block (one JVM)."""
    chat = 'true' if mode == 'chat' else 'false'
    return '\n'.join([
        f'Games={block["games"]}', 'Log=false', f'ChatNegotiation={chat}', 'FullyObservable=false',
        'ObservableVP=false', '~', block['name'], f'Trades={chat}',
        f'Agent=1,{OUR_NAME},{OUR_FACTORY}', agent_line(block['opponent'], agents), '']) + '\n'


def jvm_props(stac: dict) -> list[str]:
    props = ['-Dstac.catanai.noPlayerTrades=Y'] if trading_mode(stac) == 'off' else []
    props.append(f'-Dstac.catanai.forceEndTurnSec={int(stac.get("force_end_turn_sec", 30))}')
    if stac.get('seed') is not None:
        props.append(f'-Dstac.catanai.seed={stac["seed"]}')
    return props


# ------------------------------------------------------------------ log parsing

_KV = re.compile(r'(\w+)=(\S*)')


def server_facts(log_path: Path) -> dict[str, dict]:
    """Per-game facts printed by the patched STAC server (``catanai-*`` lines)."""
    games: dict[str, dict] = {}
    config = None

    def g(name):
        return games.setdefault(name, {'dice': [], 'mcts': [], 'force_ends': [], 'trade_blocked': 0})

    if not log_path.exists():
        return {'_config': None}
    for line in log_path.read_text(errors='replace').splitlines():
        if not line.startswith('catanai-'):
            continue
        tag = line.split(' ', 1)[0]
        kv = dict(_KV.findall(line))
        if tag == 'catanai-config':
            config = kv
        elif tag == 'catanai-seed':
            g(kv['game'])['game_seed_logged'] = int(kv['game_seed'])
        elif tag == 'catanai-start':
            row = g(kv['game'])
            row['first_player'] = int(kv['first'])
            row['seats'] = line.split(' seats=', 1)[1].split(' hexes=', 1)[0].split('|')
            for key in ('hexes', 'numbers', 'ports', 'deck'):
                m = re.search(rf' {key}=(\[[^\]]*\]|null)', line)
                row[key] = m.group(1) if m else None
        elif tag == 'catanai-observability':
            g(kv['game'])['observability'] = {k: kv.get(k) for k in ('fully_observable', 'observable_vp',
                                                                       'chat_negotiations', 'no_player_trades')}
        elif tag == 'catanai-dice':
            g(kv['game'])['dice'].append(f"{kv['roll']}:{kv['turn']}:{kv['pn']}:{kv['dice']}")
        elif tag == 'catanai-end':
            row = g(kv['game'])
            row['end'] = {'winner': int(kv['winner']), 'first': int(kv['first']), 'turns': int(kv['turns']),
                          'rounds': int(kv['rounds']), 'vp': [int(x) for x in kv['vp'].split(',')],
                          'seats': line.split(' seats=', 1)[1].split('|')}
        elif tag == 'catanai-force-end':
            g(kv['game'])['force_ends'].append({'pn': int(kv['pn']), 'state': int(kv['state']),
                                                'turn': int(kv['turn'])})
        elif tag == 'catanai-trade-blocked':
            g(kv['game'])['trade_blocked'] += 1
        elif tag == 'catanai-mcts':
            g(kv['game'])['mcts'].append({'pn': int(kv['pn']), 'ms': int(kv['ms']), 'iterations': int(kv['iterations']),
                                          'max': int(kv['max']), 'cap_ms': int(kv['cap_ms']),
                                          'cap_hit': kv['cap_hit'] == 'true'})
    return {'_config': config, **games}


def results_rows(block_dir: Path) -> dict[str, dict]:
    """STAC's own per-game results (``results/*/results.txt``): name, winner flag, VP per player."""
    out = {}
    for f in sorted((block_dir / 'results').glob('*/results.txt')):
        rows = [line.rstrip('\n').split('\t') for line in f.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        header = rows[0]
        for r in rows[1:]:
            rec = dict(zip(header, r))
            players = []
            for i in range(4):
                players.append({'name': rec.get(f'Name{i}'), 'winner': rec.get(f'Winner{i}') == '1',
                                'vp': int(rec.get(f'VP{i}') or 0),
                                'successful_offers': int(rec.get(f'SuccessfulOffers{i}') or 0)})
            out[rec['Game name']] = {'players': players, 'file': str(f)}
    return out


def summarize_block(block: dict, block_dir: Path, base_seed, bridge_seed: int) -> list[dict]:
    rows = read_jsonl(block_dir / 'decisions.jsonl')
    srv = server_facts(block_dir / 'run.log')
    stac_rows = results_rows(block_dir)
    seqs = js.decision_sequences(rows)
    games = []
    for i in range(block['games']):
        name = f'{block["name"]}_{i}'
        s = srv.get(name, {'dice': [], 'mcts': [], 'force_ends': [], 'trade_blocked': 0})
        mine = [r for r in rows if r.get('game') == name]
        start = next((r for r in mine if r.get('type') == 'game_start'), {})
        decisions = [r for r in mine if r.get('type') == 'decision']
        fallbacks = [r for r in mine if r.get('type') == 'fallback']
        chat = Counter(r.get('event') for r in mine if r.get('type') == 'chat_trade')
        errors = [r for r in decisions if r.get('cmd') == 'error']
        lat = [r['latency_ms'] for r in decisions]
        think = [r['think_ms'] for r in decisions if 'think_ms' in r]
        end = s.get('end')
        seats = s.get('seats')
        seat = start.get('seat')
        g = {'game': name, 'block': block['name'], 'opponent': block['opponent'],
             'finished': end is not None, 'seat': seat, 'seats': seats,
             'first_player': s.get('first_player'),
             'policy_seed': js.policy_seed(bridge_seed, name),
             'observability': s.get('observability'),
             'decisions': dict(Counter(r['decision'] for r in decisions)),
             'decision_count': len(decisions),
             'fallbacks': [{k: r.get(k) for k in ('decision', 'reason', 'turn_count', 'action')} for r in fallbacks],
             'fallbacks_by_type': dict(Counter(r['decision'] for r in fallbacks)),
             'bridge_errors': len(errors),
             'errors': [{k: r.get(k) for k in ('decision', 'error', 'turn_count')} for r in errors],
             'legal_mismatch_decisions': sum(bool(r.get('legal_mismatches')) for r in decisions),
             'card_plays': dict(Counter(r['action']['card'] for r in decisions
                                        if r.get('cmd') == 'play_dev' and isinstance(r.get('action'), dict))),
             'latency_ms': {'p50': percentile(lat, .5), 'p95': percentile(lat, .95),
                            'max': round(max(lat), 2) if lat else None, 'n': len(lat)},
             'think_ms_p50': percentile(think, .5),
             'dice': s['dice'], 'dice_sha': seq_sha(s['dice']),
             'deck': s.get('deck'),
             'board_sha': seq_sha([s.get('hexes'), s.get('numbers'), s.get('ports')]) if s.get('hexes') else None,
             'force_ends': s['force_ends'], 'trade_blocked': s['trade_blocked'],
             'chat_trades': dict(chat),
             'decision_seq_sha': seq_sha(seqs.get(name, [])), 'decision_seq_len': len(seqs.get(name, []))}
        searches = s['mcts']
        ms = [x['ms'] for x in searches]
        g['mcts'] = {'searches': len(searches), 'cap_hits': sum(x['cap_hit'] for x in searches),
                     # The cap is checked between iterations, so one long iteration can overrun it.
                     'cap_overruns_1s': sum(x['cap_ms'] > 0 and x['ms'] > x['cap_ms'] + 1000 for x in searches),
                     'ms_p50': percentile(ms, .5), 'ms_p95': percentile(ms, .95), 'ms_max': max(ms) if ms else None}
        if base_seed is not None:
            g['game_seed'] = derive_seed(base_seed, name, 'game')
            g['seed_logged_matches'] = s.get('game_seed_logged') == g['game_seed']
        if end is not None:
            g.update(winner=end['winner'], turns=end['turns'], rounds=end['rounds'], vp=end['vp'])
            g['won'] = (seat is not None) and end['winner'] == seat
            g['our_vp'] = end['vp'][seat] if seat is not None else None
            g['our_position'] = (seat - end['first']) % 4 if seat is not None else None
            stac_row = stac_rows.get(name)
            if stac_row is not None:
                winners = [p['name'] for p in stac_row['players'] if p['winner']]
                g['stac_results_winner'] = winners
                g['stac_results_agree'] = (winners == [end['seats'][end['winner']]]) if end['winner'] >= 0 else not winners
                g['player_trades_executed'] = sum(p['successful_offers'] for p in stac_row['players'])
            else:
                g['stac_results_agree'] = False
        games.append(g)
    return games


def aggregate(games: list[dict], expected: int, mode: str = 'off') -> dict:
    done = [g for g in games if g['finished']]
    wins = sum(bool(g.get('won')) for g in done)
    fallbacks = [dict(f, game=g['game']) for g in games for f in g['fallbacks']]
    searches = sum(g['mcts']['searches'] for g in games)
    return {
        'games': len(games), 'games_completed': len(done), 'truncated': expected - len(done),
        'our_wins': wins, 'win_rate': round(wins / len(done), 4) if done else None,
        'win_rate_wilson95': wilson(wins, len(done)),
        'seat_counts': dict(Counter(g['seat'] for g in done)),
        'position_counts': dict(Counter(g.get('our_position') for g in done)),
        'mean_our_vp': round(sum(g['our_vp'] for g in done) / len(done), 2) if done else None,
        'bridge_errors': sum(g['bridge_errors'] for g in games),
        'fallbacks_total': len(fallbacks),
        'fallbacks_by_type': dict(Counter(f['decision'] for f in fallbacks)),
        'fallback_reasons': dict(Counter((f.get('reason') or '')[:80] for f in fallbacks)),
        'legal_mismatch_decisions': sum(g['legal_mismatch_decisions'] for g in games),
        'decisions': sum(g['decision_count'] for g in games),
        'decisions_by_type': dict(sum((Counter(g['decisions']) for g in games), Counter())),
        'card_plays': dict(sum((Counter(g['card_plays']) for g in games), Counter())),
        'force_ends': sum(len(g['force_ends']) for g in games),
        'force_ends_ours': sum(1 for g in games for f in g['force_ends'] if f['pn'] == g['seat']),
        'trade_blocked': sum(g['trade_blocked'] for g in games),
        'player_trades_executed': sum(g.get('player_trades_executed', 0) for g in done),
        'chat_trades': dict(sum((Counter(g.get('chat_trades') or {}) for g in games), Counter())),
        'mcts_searches': searches, 'mcts_cap_hits': sum(g['mcts']['cap_hits'] for g in games),
        'mcts_cap_overruns_1s': sum(g['mcts']['cap_overruns_1s'] for g in games),
        'observability_violations': [g['game'] for g in done if not observability_ok(g, mode)],
        'seed_mismatches': [g['game'] for g in games if g.get('seed_logged_matches') is False],
        'stac_results_disagree': [g['game'] for g in done if not g.get('stac_results_agree')],
    }


def observability_ok(g: dict, mode: str = 'off') -> bool:
    """Hidden-information flags as logged by the server, and the declared trading mode."""
    o = g.get('observability') or {}
    chat = mode == 'chat'
    return (o.get('fully_observable') == 'false' and o.get('observable_vp') == 'false'
            and o.get('chat_negotiations') == ('true' if chat else 'false')
            and o.get('no_player_trades') == ('false' if chat else 'true'))


def latency(block_dirs: list[Path]) -> dict:
    lat, think, ms = [], [], []
    for d in block_dirs:
        for r in read_jsonl(d / 'decisions.jsonl'):
            if r.get('type') == 'decision':
                lat.append(r['latency_ms'])
                if 'think_ms' in r:
                    think.append(r['think_ms'])
        for g, s in server_facts(d / 'run.log').items():
            if not g.startswith('_'):
                ms.extend(x['ms'] for x in s['mcts'])
    stat = lambda v: {'p50': percentile(v, .5), 'p95': percentile(v, .95), 'max': max(v) if v else None, 'n': len(v)}  # noqa: E731
    return {'our_decision_ms': stat(lat), 'our_think_ms': stat(think), 'stac_mcts_search_ms': stat(ms)}


# ------------------------------------------------------------------ identity

def bridge_identity(policy: str) -> dict:
    from catan_ai.engine import BACKEND
    from catan_ai.parameter_planner import load_parameters
    files = {str(p.relative_to(ROOT)): stac_build.sha256(p)
             for p in [ROOT / 'scripts' / 'stac_bridge.py',
                       *sorted((ROOT / 'tools' / 'stac' / 'stacbridge').glob('*.py')),
                       *sorted((ROOT / 'tools' / 'jsettlers' / 'jsbridge').glob('*.py'))]}
    _, model_hash, path = load_parameters()
    return {'policy': policy, 'files': files, 'files_digest': stac_build.digest(files),
            'checkpoint': str(path), 'checkpoint_model_hash': model_hash, 'engine_backend': BACKEND,
            'python': sys.executable, 'python_version': platform.python_version()}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ run

def run_block(block: dict, out: Path, build: dict, config: dict, stac: dict) -> dict:
    block_dir = out / block['name']
    block_dir.mkdir()
    sim = block_dir / 'simulation.txt'
    sim.write_text(simulation_config(block, config['agents'], trading_mode(stac)))
    shutil.copy2(build['pdf_file'], block_dir / stac_build.PDF_FILE)
    bridge_cmd = ' '.join([sys.executable, str(ROOT / 'scripts' / 'stac_bridge.py'), '--policy', config['policy'],
                           '--seed', str(config['seed']), '--log', str(block_dir / 'bridge.jsonl'),
                           *(['--player-trades'] if trading_mode(stac) == 'chat' else [])])
    cmd = [stac_build.java(), f'-Xmx{block.get("jvm_heap", stac.get("jvm_heap", "4g"))}', '-cp', build['classpath'],
           *jvm_props(stac),
           f'-Dcatanai.bridge.cmd={bridge_cmd}', f'-Dcatanai.bridge.cwd={ROOT}',
           f'-Dcatanai.bridge.stderr={block_dir / "bridge.stderr"}',
           f'-Dcatanai.decision.log={block_dir / "decisions.jsonl"}',
           f'-Dcatanai.bridge.timeout_ms={int(config.get("bridge_timeout_ms", 30000))}',
           'soc.robot.stac.simulation.Simulation', sim.name]
    (block_dir / 'command.json').write_text(json.dumps(cmd, indent=1) + '\n')
    started = time.time()
    rc, timed_out = supervise(cmd, block_dir, block_dir / 'run.log',
                              config['timeout_sec_per_game'] * block['games'], env=dict(os.environ))
    return {'block': block['name'], 'rc': rc, 'timed_out': timed_out, 'elapsed_sec': round(time.time() - started, 1),
            'simulation_config_sha256': stac_build.sha256(sim)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--out', help='new output directory (default runs/stac-match/<utc stamp>); must not exist')
    parser.add_argument('--compare', nargs=2, metavar=('RECEIPT_A', 'RECEIPT_B'),
                        help='compare two seeded receipts game by game and exit')
    args = parser.parse_args(argv)
    if args.compare:
        result = compare_receipts(*args.compare)
        print(json.dumps(result, indent=1))
        return 0 if result['compatible'] and result['incomplete_games'] == 0 else 1
    if not args.config:
        parser.error('--config is required')
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    stac = check_config(config)
    mode = trading_mode(stac)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = allocate_out(Path(args.out) if args.out else ROOT / 'runs' / 'stac-match' / stamp)
    build = stac_build.build_manifest(out / 'build', stac['patch'])
    write_exclusive(out / 'build-manifest.json', build)
    blocks = config['blocks']
    manifest = {
        'name': config['name'], 'created_utc': stamp, 'config_path': str(config_path),
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'config': config,
        'games_requested': sum(b['games'] for b in blocks),
        'source_revision': git('rev-parse', 'HEAD'),
        'source_dirty': bool(git('status', '--porcelain', '--', 'tools', 'scripts', 'src', 'checkpoints', 'experiments')),
        'build': {'manifest_sha256': stac_build.sha256(out / 'build-manifest.json'),
                  'stac': {k: v for k, v in build['stac'].items() if k != 'files'},
                  'jdk_version': build['jdk']['version'].splitlines()[0],
                  'bridge_java_sources_digest': build['bridge_java']['sources_digest'],
                  'bridge_java_classes_digest': build['bridge_java']['classes_digest']},
        'bridge': bridge_identity(config['policy']),
        'stac': {'seed': stac.get('seed'), 'patch_sha256': build['stac']['patch_sha256'],
                 'jvm_props': jvm_props(stac), 'build_dir': build['build_dir'], 'player_trading': mode},
    }
    write_exclusive(out / 'manifest.json', manifest)
    stac_build.verify_loaded(build)  # nothing on the class path changed since verification
    started = time.time()
    with ThreadPoolExecutor(max_workers=int(config.get('parallel_blocks', 1))) as pool:
        runs = list(pool.map(lambda b: run_block(b, out, build, config, stac), blocks))
    elapsed = round(time.time() - started, 1)
    per_block, all_games = {}, []
    for b in blocks:
        games = summarize_block(b, out / b['name'], stac.get('seed'), config['seed'])
        per_block[b['name']] = {'opponent': b['opponent'], **aggregate(games, b['games'], mode)}
        all_games.extend(games)
    retired = [r for b in blocks for r in read_jsonl(out / b['name'] / 'decisions.jsonl')
               if r.get('type') == 'bridge_retired']
    total = aggregate(all_games, manifest['games_requested'], mode)
    bar = config.get('pass_bar', {})
    receipt = {
        **{k: manifest[k] for k in ('name', 'created_utc', 'config_path', 'config_sha256', 'config',
                                    'games_requested', 'source_revision', 'source_dirty', 'build', 'bridge', 'stac')},
        'manifest_sha256': stac_build.sha256(out / 'manifest.json'),
        'runs': runs, 'elapsed_sec': elapsed, 'bridge_retired': retired,
        'latency': latency([out / b['name'] for b in blocks]),
        'blocks': per_block, 'total': total, 'games': all_games,
        'artifacts': {b['name']: str(out / b['name']) for b in blocks},
    }
    receipt['pass'] = (total['games_completed'] >= bar.get('completed_games', manifest['games_requested'])
                       and total['bridge_errors'] <= bar.get('bridge_errors', 0)
                       and not retired and not any(r['timed_out'] for r in runs)
                       and not total['observability_violations'] and not total['seed_mismatches']
                       and (mode == 'chat' or total['player_trades_executed'] == 0)
                       and total['player_trades_executed'] >= bar.get('min_player_trades_executed', 0)
                       and total['chat_trades'].get('offer_timeout', 0) <= bar.get('max_offer_timeouts', 10**9)
                       and total['force_ends'] <= bar.get('max_force_ends', 10**9)
                       and not total['stac_results_disagree'])
    write_exclusive(out / 'receipt.json', receipt)
    print(json.dumps({'pass': receipt['pass'], 'elapsed_sec': elapsed,
                      **{n: {k: v[k] for k in ('games_completed', 'our_wins', 'win_rate_wilson95', 'bridge_errors',
                                                'fallbacks_by_type', 'mcts_cap_hits', 'mcts_cap_overruns_1s',
                                                'force_ends', 'player_trades_executed', 'chat_trades')}
                         for n, v in per_block.items()}}))
    print(f'receipt: {out / "receipt.json"}')
    return 0 if receipt['pass'] else 1


# ------------------------------------------------------------------ determinism compare

SETUP_KEYS = ('board_sha', 'deck', 'first_player', 'seats', 'seat')
OUTCOME_KEYS = ('dice_sha', 'winner', 'turns', 'decision_seq_sha')
COMPAT_KEYS = (('stac', 'seed'), ('stac', 'patch_sha256'), ('config', 'policy'), ('config', 'seed'),
               ('config', 'blocks'), ('config', 'agents'), ('games_requested',))


def _missing(g: dict, decisions: list) -> list[str]:
    missing = [k for k in SETUP_KEYS + OUTCOME_KEYS if g.get(k) in (None, '', [])]
    if not g.get('finished'):
        missing.append('finished')
    if not g.get('dice'):
        missing.append('dice (no rolls logged)')
    if not decisions:
        missing.append('decisions (none logged)')
    elif seq_sha(decisions) != g.get('decision_seq_sha'):
        missing.append('decisions (log does not match receipt decision_seq_sha)')
    return missing


def compare_receipts(path_a, path_b) -> dict:
    """Game-by-game comparison of two seeded receipts; fails closed on absent evidence."""
    ra, rb = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    problems = []
    for path in COMPAT_KEYS:
        va, vb = js._dig(ra, path), js._dig(rb, path)
        if va != vb:
            problems.append(f'{".".join(path)} differs')
    if js._dig(ra, ('stac', 'seed')) is None:
        problems.append('receipt is not a seeded run (stac.seed missing)')
    names_a = [g.get('game') for g in ra.get('games') or []]
    names_b = [g.get('game') for g in rb.get('games') or []]
    for tag, names in (('a', names_a), ('b', names_b)):
        if not names or len(set(names)) != len(names) or None in names:
            problems.append(f'receipt {tag} has no, missing or duplicate game names')
    if set(names_a) != set(names_b):
        problems.append('game sets differ')

    def seqs(r):
        out = {}
        for d in (r.get('artifacts') or {}).values():
            out.update(js.decision_sequences(read_jsonl(Path(d) / 'decisions.jsonl')))
        return out

    sa_all, sb_all = seqs(ra), seqs(rb)
    gb = {g.get('game'): g for g in rb.get('games') or []}
    rows = []
    for a in ra.get('games') or []:
        name = a.get('game')
        b = gb.get(name)
        if b is None:
            rows.append({'game': name, 'incomplete': ['missing in b']})
            continue
        sa, sb = sa_all.get(name, []), sb_all.get(name, [])
        missing = [f'a: {m}' for m in _missing(a, sa)] + [f'b: {m}' for m in _missing(b, sb)]
        if missing:
            rows.append({'game': name, 'incomplete': missing})
            continue
        setup = {k: a[k] == b[k] for k in SETUP_KEYS}
        va, vb = ([d.rsplit(':', 1)[-1] for d in g['dice']] for g in (a, b))
        n = min(len(va), len(vb))
        dec_prefix = js._common_prefix(sa, sb)
        row = {'game': name, 'block': a.get('block'), 'setup_identical': all(setup.values()),
               'setup_diff': [k for k, v in setup.items() if not v],
               'dice': [len(a['dice']), len(b['dice'])],
               'dice_values_identical_for_every_k': va[:n] == vb[:n],
               'dice_values_common_prefix': js._common_prefix(va, vb),
               'decisions': [len(sa), len(sb)], 'decision_common_prefix': dec_prefix,
               'winner': [a['winner'], b['winner']], 'turns': [a['turns'], b['turns']]}
        row['end_to_end_identical'] = row['setup_identical'] and all(a[k] == b[k] for k in OUTCOME_KEYS)
        if not row['end_to_end_identical'] and dec_prefix < max(len(sa), len(sb)):
            row['first_decision_divergence'] = {'index': dec_prefix,
                                                'a': sa[dec_prefix] if dec_prefix < len(sa) else None,
                                                'b': sb[dec_prefix] if dec_prefix < len(sb) else None}
        rows.append(row)
    complete = [r for r in rows if 'incomplete' not in r] if not problems else []
    return {'a': str(path_a), 'b': str(path_b), 'compatible': not problems, 'problems': problems,
            'games': len(rows), 'complete_games': len(complete), 'incomplete_games': len(rows) - len(complete),
            'setup_identical': sum(r['setup_identical'] for r in complete),
            'setup_and_dice_k_identical': sum(r['setup_identical'] and r['dice_values_identical_for_every_k']
                                              for r in complete),
            'end_to_end_identical': sum(r['end_to_end_identical'] for r in complete),
            'same_winner': sum(r['winner'][0] == r['winner'][1] for r in complete),
            'rows': rows}


if __name__ == '__main__':
    raise SystemExit(main())
