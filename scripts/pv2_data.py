"""Counterfactual afterstate data for the PV v2 value model (model audit fixes 1 and 5).

At sampled own-turn decisions of self-play games of the served policy, the
candidates are exactly what PV search scores (``catan_ai.pv_search``): "end the
turn now" plus the ``k`` best served sequences by first move. For each candidate:

- ``x``: encoder-v2 features of the afterstate from the chooser's information only
  (``afterstate_encoding``; a purchase is averaged over the deck prior);
- ``wins/playouts``: the candidate is applied to a copy of the TRUE game, the turn
  is ended, and the game is played to the end by copies of the served bots, with
  a fresh dice stream and a reshuffled remaining deck per playout; playout ``r``
  uses the same streams for every candidate (common random numbers).

Exploration: with probability ``explore`` the real game then follows a uniformly
chosen candidate (sequence, then end of turn) instead of the served move, so the
visited positions include held cards and non-served builds. Test-split games use
``playouts_test`` playouts per candidate for a precise within-decision gate.

Fail-closed manifest and per-game identity as in ``rank_data.py``; per-game files
are written with ``open('x')``.
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
from rank_data import bind_manifest, canonical_sha, source_revision  # noqa: E402

from catan_ai.experiments import source_files  # noqa: E402


def run_identity(config):
    from catan_ai.bots import make_policy
    from catan_ai.encoding import size
    (tlo, tn), (vlo, vn) = config['split']['train'], config['split']['test']
    if set(range(tlo, tlo + tn)) & set(range(vlo, vlo + vn)):
        raise ValueError('train and test seeds overlap')
    identity = {'config': config, 'config_sha256': canonical_sha(config), 'source': source_files(),
                'source_revision': source_revision(), 'encoding_size': size(),
                'checkpoint_sha256': getattr(make_policy(config['policy'], 0), 'model_hash', None),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity['identity_sha256'] = canonical_sha(identity)
    return identity


def apply_turn(game, seat, seq):
    """Apply ``seq`` then end the turn on ``game``; ``False`` if any step is illegal."""
    for action in seq:
        if action not in game.legal_actions(seat):
            return False
        game.apply(seat, action)
        if game.winner is not None:
            return True
    if game.winner is None:
        end = {'type': 'end_turn'}
        if end not in game.legal_actions(seat):
            return False
        game.apply(seat, end)
    return True


def playout(game, bots, seat, seq, stream, max_turns):
    from catan_ai.observation import Observation
    game = deepcopy(game)
    bots = [deepcopy(b) for b in bots]
    game.rng = random.Random(stream)
    game.deck = list(game.deck)
    game.rng.shuffle(game.deck)
    if not apply_turn(game, seat, seq):
        return None
    while game.winner is None and game.turn < max_turns:
        s = game.current
        game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    return game.winner


def run_game(args):
    seed, config, path, identity_sha, playouts = args
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.pv_search import afterstate_encoding, candidate_sequences
    path = Path(path)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get('seed') != seed or record.get('identity') != identity_sha:
            raise ValueError(f'record for seed {seed} does not match this run identity')
        return 'exists'
    started = time.monotonic()
    policy = config['policy']
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = [make_policy(policy, seed * 17 + i + 1) for i in range(4)]
    pick = random.Random(seed * 7919 + 3)
    decisions, seen, failed = [], set(), 0
    while game.winner is None and game.turn < config['max_turns']:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        first = game.phase == 'actions' and (game.turn, seat) not in seen
        if first:
            seen.add((game.turn, seat))
        sampled = first and any(a['type'] == 'end_turn' for a in view.actions) and pick.random() < config['decision_rate']
        if sampled:
            probe = deepcopy(bots[seat])
            served = bots[seat].choose(view, seat)
            seqs = candidate_sequences(probe, view, seat, config['candidates'])
            rows = []
            for seq in seqs:
                encoded = afterstate_encoding(probe, view, seat, seq)
                if encoded is None:
                    continue
                x, won_now = encoded
                row = {'seq': seq, 'served': bool(seq) and seq[0] == served or (not seq and served['type'] in ('end_turn', 'offer_trade')),
                       'x': x, 'won_now': won_now, 'wins': 0, 'playouts': 0, 'illegal': 0}
                for r in range(playouts):
                    winner = playout(game, bots[:seat] + [probe] + bots[seat + 1:], seat, seq,
                                     seed * 1_000_003 + game.turn * 101 + r, config['max_turns'])
                    if winner is None and not apply_turn(deepcopy(game), seat, seq):
                        row['illegal'] += 1
                        continue
                    row['wins'] += winner == seat
                    row['playouts'] += 1
                rows.append(row)
            if len(rows) >= 2:
                decisions.append({'turn': game.turn, 'seat': seat, 'candidates': rows})
            if rows and pick.random() < config['explore']:
                choice = rows[pick.randrange(len(rows))]['seq']
                if apply_turn(game, seat, choice):
                    continue
                failed += 1
            game.apply(seat, served)
            continue
        game.apply(seat, bots[seat].choose(view, seat))
    record = {'seed': seed, 'identity': identity_sha, 'schema': 'pv2', 'winner': game.winner, 'turns': game.turn,
              'truncated_game': game.winner is None, 'explore_failed': failed, 'playouts': playouts,
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
    config = json.loads(args.config.read_text())
    identity = bind_manifest(args.root, run_identity(config))
    directory = args.root / 'games'
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, key in (('train', 'playouts'), ('test', 'playouts_test')):
        lo, n = config['split'][split]
        jobs += [(seed, config, str(directory / f'{seed}.json'), identity['identity_sha256'], config[key])
                 for seed in range(lo, lo + n)]
    started, done = time.monotonic(), 0
    with Pool(args.workers) as pool:
        for _status in pool.imap_unordered(run_game, jobs):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(json.dumps({'games': done, 'of': len(jobs), 'seconds': round(time.monotonic() - started)}),
                      flush=True)


if __name__ == '__main__':
    main()
