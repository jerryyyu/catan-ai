"""External ladder: our bot alone against three Catanatron players of one variant.

Descriptive comparison (Jerry 2026-11-07: "try play vs stronger"; "retire when we beat
them too easily"). For each seed the candidate takes each seat once against three
copies of the variant (``ExternalSearchBot``, belief-sampled hidden state, no cheating).
Per game: winner, turns, external fallbacks/budget failures, decision times. The
fair-share win rate is .25.

Run identity (Codex HOLD @31b6376): ``main`` binds the root to an immutable
``manifest.json`` (``ladder_identity``: config, this script's and its helper's SHA-256,
``source_files()`` including the engine backend, the source revision resolved from this
repository (with its dirty flag, or an explicit ``git_metadata: unavailable`` marker and
the snapshot's ``COMMIT`` file), the candidate's model hash and value-net weights/meta
hashes, and each declared variant's verified Catanatron install), written with
``open('x')``. It refuses a root whose manifest differs, a non-empty root without one,
or a stored game that does not match its job, before scheduling any game. Each game
record carries its seed, seat, variant and the manifest's ``run_identity``; ``play``
reuses a record only when all four match and refuses otherwise. A changed producer or
config needs a new root; completed records are never rewritten.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from rank_data import canonical_sha  # noqa: E402


def repo_revision(root=REPO):
    """Git revision and dirty flag of ``root`` (not the caller's CWD); an explicit marker when unavailable."""
    root = Path(root).resolve()

    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout

    try:
        if Path(git('rev-parse', '--show-toplevel').strip()).resolve() != root:
            raise ValueError('not the repository root')
        revision = git('rev-parse', 'HEAD').strip()
        status = git('status', '--porcelain', '--untracked-files=no')
        return {'git_metadata': 'available', 'revision': revision, 'dirty': bool(status.strip()),
                'dirty_sha256': hashlib.sha256(status.encode()).hexdigest() if status.strip() else None}
    except (OSError, subprocess.CalledProcessError, ValueError):
        commit = root / 'COMMIT'
        return {'git_metadata': 'unavailable', 'revision': None, 'dirty': None,
                'commit_file': commit.read_text().strip() if commit.exists() else None}


def candidate_identity(name):
    """Model identity of the candidate as served: model hash and verified value-net weights."""
    from catan_ai.bots import make_policy
    bot = make_policy(name, 0)
    out = {'name': name, 'class': type(bot).__name__, 'model_hash': getattr(bot, 'model_hash', None)}
    net = getattr(bot, 'net', None)
    meta = getattr(net, 'meta', None) if net is not None else None
    if net is None:
        out.update(value_net=None, value_net_reason='policy has no value net')
    elif not isinstance(meta, dict) or not meta.get('npz_sha256'):
        out.update(value_net=None, value_net_reason=f'{type(net).__name__} exposes no weights metadata')
    else:
        # The nets refuse to load unless the .npz hashes to meta['npz_sha256'], so this is verified.
        out['value_net'] = {'class': type(net).__name__, 'npz_sha256': meta['npz_sha256'],
                            'meta_sha256': canonical_sha(meta)}
    return out


def external_identity(variant, python=None):
    """``external_worker.verify_install(variant)`` run in the comparator's interpreter (as the worker does)."""
    python = python or os.environ.get('CATAN_EXTERNAL_PYTHON', str(REPO / 'runs/external/catanatron-env/bin/python'))
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'src'), 'PYTHONHASHSEED': '0'}
    code = 'import json, sys; from catan_ai.external_worker import verify_install; print(json.dumps(verify_install(sys.argv[1])))'
    out = subprocess.run([python, '-c', code, variant], capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout)


def ladder_identity(cfg):
    """The immutable identity of a ladder run: declaration, producer code, engine, candidate and comparators."""
    from catan_ai.engine import BACKEND
    from catan_ai.experiments import source_files
    scripts = sorted({Path(__file__).resolve(), HERE / 'rank_data.py'})
    identity = {'config': cfg, 'config_sha256': canonical_sha(cfg),
                'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
                'source': source_files(), 'engine_backend': BACKEND, 'source_revision': repo_revision(),
                'candidate': candidate_identity(cfg['candidate']),
                'external': {v: external_identity(v) for v in sorted(cfg['variants'])}}
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


def check_record(path, seed, seat, variant, identity_sha):
    """The stored record at ``path``, refused unless it is this seed, seat, variant and run identity."""
    record = json.loads(Path(path).read_text())
    got = (record.get('seed'), record.get('seat'), record.get('variant'), record.get('run_identity'))
    if got != (seed, seat, variant, identity_sha):
        raise ValueError(f'{path} (seed {got[0]}, seat {got[1]}, variant {got[2]}, identity {got[3]}) does not match '
                         f'seed {seed}, seat {seat}, variant {variant}, identity {identity_sha}')
    return record


def play(job):
    if len(job) != 8:
        raise ValueError(f'job {job!r} lacks the run identity (seed, seat, variant, candidate, budget, max_turns, path, identity)')
    seed, seat, variant, candidate, budget, max_turns, path, identity_sha = job
    path = Path(path)
    if path.exists():
        return check_record(path, seed, seat, variant, identity_sha)
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.external_policy import ExternalSearchBot
    from catan_ai.observation import Observation
    started = time.monotonic()
    game = Game.new([f'p{i}' for i in range(4)], seed)
    bots = [make_policy(candidate, seed * 17 + i + 1) if i == seat else
            ExternalSearchBot(seed * 1000 + i, seconds=budget['decision'], turn_seconds=budget['turn'], variant=variant)
            for i in range(4)]
    try:
        while game.winner is None and game.turn < max_turns:
            s = game.current
            game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    finally:
        for i, b in enumerate(bots):
            if i != seat:
                b.close()
    ext = [d for i, b in enumerate(bots) if i != seat for d in b.diagnostics]
    record = {'seed': seed, 'seat': seat, 'variant': variant, 'run_identity': identity_sha,
              'winner': game.winner, 'turns': game.turn,
              'truncated': game.winner is None, 'seconds': time.monotonic() - started,
              'external_decisions': len(ext), 'external_fallbacks': sum(d['fallback'] for d in ext),
              'external_budget_failures': sum(d['budget_failure'] for d in ext),
              'external_errors': sorted({d['error'] for d in ext if d['error']})[:5],
              'external_seconds_p95': sorted(d['seconds'] for d in ext)[int(.95 * len(ext))] if ext else None,
              'identity': next((b.identity for i, b in enumerate(bots) if i != seat and b.identity), None),
              'external_identities': sorted({canonical_sha(b.identity) for i, b in enumerate(bots)
                                             if i != seat and b.identity})}
    with path.open('x') as stream:
        json.dump(record, stream)
    return record


def build_jobs(cfg, root, variants, identity_sha):
    """Every declared game of ``variants``; refuse stored games that are undeclared or do not match."""
    root = Path(root)
    jobs = []
    for variant in variants:
        if variant not in cfg['variants']:
            raise ValueError(f'variant {variant} is not declared in the config')
        spec = cfg['variants'][variant]
        jobs += [(seed, seat, variant, cfg['candidate'], spec['budget'], cfg['max_turns'],
                  str(root / variant / f'{seed}-{seat}.json'), identity_sha)
                 for seed in range(spec['seed'], spec['seed'] + spec['seeds']) for seat in range(4)]
    declared = {job[6] for job in jobs}
    for entry in sorted(root.iterdir()) if root.exists() else []:
        if entry.name != 'manifest.json' and entry.name not in cfg['variants']:
            raise ValueError(f'{entry} is not part of this run (declared variants {sorted(cfg["variants"])}); use a new root')
    for variant in variants:
        for stored in sorted((root / variant).glob('*')) if (root / variant).exists() else []:
            if str(stored) not in declared:
                raise ValueError(f'{stored} is not a declared game of this run; use a new root')
    for job in jobs:
        if Path(job[6]).exists():
            check_record(job[6], job[0], job[1], job[2], identity_sha)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--variants', nargs='*', help='subset of the config variants (default all)')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    identity = ladder_identity(cfg)
    prepare_root(args.root, identity)
    identity_sha = identity['identity_sha256']
    jobs = build_jobs(cfg, args.root, args.variants or list(cfg['variants']), identity_sha)
    for variant in {job[2] for job in jobs}:
        (args.root / variant).mkdir(exist_ok=True)
    print(json.dumps({'root': str(args.root), 'identity_sha256': identity_sha, 'host': platform.node(),
                      'games': len(jobs), 'stored': sum(Path(j[6]).exists() for j in jobs)}), flush=True)
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(play, jobs):
            done += 1
            print(json.dumps({'done': done, 'of': len(jobs), 'variant': r['variant'], 'won': r['winner'] == r['seat'],
                              'fallbacks': r['external_fallbacks'], 'seconds': round(time.monotonic() - started)}), flush=True)


if __name__ == '__main__':
    main()
