"""Search diagnostic: where does our planning leave wins? (Jerry 2026-11-06; evaluation, not training data.)

At sampled own-turn decisions of the PV baseline (``planner_pv2_b_fix``, seat 0) against
a declared opponent lineup, every option is evaluated by paired playouts from the TRUE
game (each playout ``r`` uses the same dice stream and reshuffled remaining deck for every
option; the same opponents as the lineup; the served bot finishes our seat's game):

- ``first``: every legal first move of the turn (builds, bank trades, development purchase,
  Year of Plenty / Road Building plays, end turn), after which the served bot plays on;
- ``seq``: the served search's candidate sequences ("end now" + top ``k`` by first move),
  each applied and followed by ending the turn (what PV search chooses among).

Per option the per-playout outcomes are stored (bits), so the analysis can split each
option's playouts in two halves: choose with one half, score with the other (the gain of
a finite-sample selector, not a perfect oracle; see ``analyze_search_diagnostic``), and
measure how well the halves agree (label reliability). The decision also
records what the served bot and the PV bot chose.

Run identity (Codex HOLD @c32a17a, P2): ``main`` binds the root to an immutable
``manifest.json`` (config, this script's and its helpers' SHA-256, ``source_files()``,
source revision, the policies' model hashes), written with ``open('x')``, and refuses a
root whose manifest differs, or a non-empty root without one, before writing anything.
Each game record carries its seed, lineup and the manifest's ``identity_sha256``; resume
skips a game only when its record matches all three and refuses otherwise.
``declared_records`` (used by the analyses) refuses a root unless every declared seed of
every lineup is present, nothing undeclared is, and every record matches the manifest.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pv2_data import apply_turn  # noqa: E402
from rank_data import canonical_sha, source_revision  # noqa: E402

HERE = Path(__file__).resolve().parent

KINDS = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development', 'end_turn')


def first_moves(view, seat):
    out = []
    for a in view.actions:
        if a['type'] in KINDS or (a['type'] == 'play_development' and a.get('card') in ('year_of_plenty', 'road_building')):
            out.append(a)
    return out


def run_playout(game, seat, opponents_cls, served_seed, option, kind, stream, max_turns):
    from catan_ai.bots import make_policy
    from catan_ai.observation import Observation
    g = deepcopy(game)
    g.rng = random.Random(stream)
    g.deck = list(g.deck)
    g.rng.shuffle(g.deck)
    bots = [make_policy(name, served_seed + i) for i, name in enumerate(opponents_cls)]
    if kind == 'seq':
        if not apply_turn(g, seat, option):
            return None
    else:
        if option not in g.legal_actions(seat):
            return None
        g.apply(seat, option)
    while g.winner is None and g.turn < max_turns:
        s = g.current
        g.apply(s, bots[s].choose(Observation.for_policy(g, s, bots[s]), s))
    return int(g.winner == seat)


def diagnostic_identity(cfg, script):
    """The immutable identity of a diagnostic run: declaration, producer code and models."""
    from catan_ai.bots import make_policy
    from catan_ai.experiments import source_files
    names = sorted({'planner_pv2_b_fix', 'planner_offer_accept', *(n for lineup in cfg['lineups'].values() for n in lineup)})
    models = {}
    for name in names:
        bot = make_policy(name, 0)
        models[name] = {'model_hash': getattr(bot, 'model_hash', None),
                        'value_net_sha256': getattr(getattr(bot, 'net', None), 'meta', {}).get('npz_sha256')}
    scripts = sorted({Path(script).resolve(), HERE / 'search_diagnostic.py', HERE / 'pv2_data.py', HERE / 'rank_data.py'})
    identity = {'config': cfg, 'config_sha256': canonical_sha(cfg), 'script': Path(script).name,
                'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
                'source': source_files(), 'source_revision': source_revision(), 'models': models}
    identity['identity_sha256'] = canonical_sha(identity)
    return identity


def prepare_root(root, identity):
    """Bind ``root`` to ``identity``; refuse a different or missing manifest before touching it."""
    root = Path(root)
    path = root / 'manifest.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('identity_sha256') != identity['identity_sha256']:
            changed = sorted(k for k in identity if existing.get(k) != identity[k] and k != 'identity_sha256')
            raise ValueError(f'{root} belongs to a different run identity (differs in {changed}); use a new root')
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'{root} has content but no manifest.json (unbound root); use a new root')
    root.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(identity, stream, indent=1)
    return identity


def check_record(path, seed, lineup, identity_sha):
    """The stored record at ``path``, refused unless it is this seed, lineup and run identity."""
    record = json.loads(Path(path).read_text())
    got = (record.get('seed'), record.get('lineup'), record.get('identity'))
    if got != (seed, lineup, identity_sha):
        raise ValueError(f'{path} (seed {got[0]}, lineup {got[1]}, identity {got[2]}) does not match '
                         f'seed {seed}, lineup {lineup}, identity {identity_sha}')
    return record


def declared_records(root, legacy=False):
    """``({lineup: [record per declared seed]}, identity)`` after verifying complete coverage.

    ``legacy=True`` admits a pre-manifest root (``config.json`` only); its records carry
    no identity, so only seeds, lineups and coverage are checked and the result is
    marked ``legacy_unbound``."""
    root = Path(root)
    path = root / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if canonical_sha({k: v for k, v in manifest.items() if k != 'identity_sha256'}) != manifest['identity_sha256']:
            raise ValueError(f'{path} does not hash to its identity')
        cfg, identity_sha = manifest['config'], manifest['identity_sha256']
    elif legacy:
        cfg, identity_sha = json.loads((root / 'config.json').read_text())['config'], None
    else:
        raise ValueError(f'{root} has no manifest.json (pass legacy=True / --legacy for a pre-manifest root)')
    lineups = {p.name for p in root.iterdir() if p.is_dir()}
    if lineups != set(cfg['seeds']):
        raise ValueError(f'lineup directories {sorted(lineups)} differ from the declared {sorted(cfg["seeds"])}')
    out = {}
    for lineup, (lo, n) in sorted(cfg['seeds'].items()):
        expected = {str(s) for s in range(lo, lo + n)}
        found = {p.stem for p in (root / lineup).glob('*.json')}
        if found != expected:
            raise ValueError(f'{lineup}: coverage differs from the declaration: missing {len(expected - found)} '
                             f'(e.g. {sorted(expected - found)[:3]}), unexpected {sorted(found - expected)[:3]}')
        out[lineup] = []
        for stem in sorted(expected):
            if identity_sha is None:
                record = json.loads((root / lineup / f'{stem}.json').read_text())
                if (record.get('seed'), record.get('lineup')) != (int(stem), lineup):
                    raise ValueError(f'{lineup}/{stem}.json is not seed {stem} of lineup {lineup}')
            else:
                record = check_record(root / lineup / f'{stem}.json', int(stem), lineup, identity_sha)
            out[lineup].append(record)
    return out, {'identity_sha256': identity_sha, 'legacy_unbound': identity_sha is None,
                 'games': {k: len(v) for k, v in out.items()}}


def run_game(args):
    seed, cfg, lineup_name, path, identity_sha = args
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.pv_search import candidate_sequences
    path = Path(path)
    if path.exists():
        check_record(path, seed, lineup_name, identity_sha)
        return 'exists'
    started = time.monotonic()
    opponents = cfg['lineups'][lineup_name]
    names = ['planner_pv2_b_fix'] + opponents
    game = Game.new([f'{n}-{i}' for i, n in enumerate(names)], seed)
    bots = [make_policy(n, seed * 17 + i + 1) for i, n in enumerate(names)]
    shadow = make_policy('planner_offer_accept', seed * 17 + 1)
    playout_cls = ['planner_offer_accept'] + opponents  # served bot finishes our seat's game
    pick = random.Random(seed * 7919 + 11)
    seen, decisions = set(), []
    while game.winner is None and game.turn < cfg['max_turns'] and len(decisions) < cfg['max_decisions_per_game']:
        s = game.current
        view = Observation.for_policy(game, s, bots[s])
        first = s == 0 and game.phase == 'actions' and (game.turn, s) not in seen
        if first:
            seen.add((game.turn, s))
        if first and any(a['type'] == 'end_turn' for a in view.actions) and pick.random() < cfg['decision_rate']:
            shadow.last_offer_turn = bots[0].last_offer_turn
            served = deepcopy(shadow).choose(view, s)
            options = [('first', a) for a in first_moves(view, s)]
            options += [('seq', q) for q in candidate_sequences(shadow, view, s, cfg['candidates'])]
            rows = []
            for kind, opt in options:
                bits = []
                for r in range(cfg['playouts']):
                    w = run_playout(game, s, playout_cls, seed * 31, opt, kind,
                                    seed * 1_000_003 + game.turn * 1009 + r, cfg['max_turns'])
                    if w is None:
                        bits = None
                        break
                    bits.append(w)
                if bits is not None:
                    rows.append({'kind': kind, 'option': opt, 'bits': bits})
            action = bots[0].choose(view, s)  # the PV baseline's actual move (also advances the game)
            decisions.append({'turn': game.turn, 'points': [p.public_points for p in game.players],
                              'served': served, 'pv': action, 'options': rows})
            game.apply(s, action)
            continue
        game.apply(s, bots[s].choose(view, s))
    record = {'seed': seed, 'lineup': lineup_name, 'identity': identity_sha, 'winner': game.winner, 'turns': game.turn,
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
    identity = prepare_root(args.root, diagnostic_identity(cfg, __file__))
    jobs = []
    for lineup_name, (lo, n) in cfg['seeds'].items():
        (args.root / lineup_name).mkdir(exist_ok=True)
        jobs += [(seed, cfg, lineup_name, str(args.root / lineup_name / f'{seed}.json'), identity['identity_sha256'])
                 for seed in range(lo, lo + n)]
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(run_game, jobs):
            done += 1
            print(json.dumps({'games': done, 'of': len(jobs), 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
