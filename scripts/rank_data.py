"""Policy/value (PV) data: playout win rates for the served bot's candidate first moves.

Schema 2 (2026-11-04, Jerry: train a shengji-style policy/value model): each
decision stores the pre-move observation (``obs``), and each candidate an action
encoding (``act``), its post-move observation (``after``) and its playout win rate,
so a shared-trunk network can learn a policy head (which candidate) and a value
head (win chance). ``value_rows`` add every turn-start position (and, when
``value_points`` includes ``end_turn``, every end-of-turn position) labelled with
the final winner, for the value head; ``point`` names which.


Data generation for the move-ranking lane (board row 4 follow-up, Jerry
2026-11-04). Not a strength claim and no model is trained here.

For each game seed, four copies of the served policy play a self-play game. At
the first action decision of an own turn, sampled with probability
``decision_rate`` from a seed-derived stream, the generator takes up to
``candidates`` first moves: the served choice plus the best others by the served
one-step leaf on the card-counted world (builds, bank trades, development
purchase, end turn). Each candidate is applied to a copy of the TRUE game and the
game is played to the end ``playouts`` times by copies of the served bots, each
playout with its own dice stream and a reshuffled remaining development deck;
playout ``r`` uses the same streams for every candidate (common random numbers).
The real game then continues with the served move.

A candidate row stores observation-only features of the seat's own view right
after the move (``value_features`` plus served leaf/shared score on the
card-counted world built from that view), the move itself, and wins/playouts.
True hidden state is used only to play the outcome, never as a feature.

Every run root has a fail-closed manifest (config, sources, revision, checkpoint,
script); every per-game file carries the identity and is written with
``open('x')``. Resume rejects a mismatched manifest or record.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import time

from catan_ai.experiments import source_files

KINDS = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development', 'end_turn')
RESOURCES = ('brick', 'lumber', 'wool', 'grain', 'ore')


def action_encoding(action):
    """16 numbers: move type one-hot, then resource given and received for a bank trade."""
    out = [float(action['type'] == kind) for kind in KINDS] + [0.] * 10
    if action['type'] == 'bank_trade':
        out[6 + RESOURCES.index(action['give'])] = 1.
        out[11 + RESOURCES.index(action['receive'])] = 1.
    return out


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_revision():
    import catan_ai
    commit = Path(catan_ai.__file__).resolve().parents[2] / 'COMMIT'
    return commit.read_text().strip() if commit.exists() else None


def run_identity(config):
    from catan_ai.bots import make_policy
    identity = {'config': config, 'config_sha256': canonical_sha(config), 'source': source_files(),
                'source_revision': source_revision(),
                'checkpoint_sha256': getattr(make_policy(config['policy'], 0), 'model_hash', None),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity['identity_sha256'] = canonical_sha(identity)
    return identity


def bind_manifest(root, identity):
    path = Path(root) / 'manifest.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('identity_sha256') != identity['identity_sha256']:
            raise ValueError('run root belongs to a different producer/config/checkpoint; use a new root')
        return existing
    Path(root).mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(identity, stream, indent=1)
    return identity


def features(game, seat, bot):
    from catan_ai.observation import Observation
    from catan_ai.value_model import value_features
    view = Observation.for_policy(game, seat, bot)
    world = bot._world(view)
    leaves = [bot._value(world, s) for s in range(4)]
    return {'features': value_features(view) + [leaves[seat] / 100,
                                                max(leaves[s] for s in range(4) if s != seat) / 100],
            'leaf': leaves[seat], 'shared': bot._score(world, seat)}


def candidates(view, seat, bot, served, limit):
    world = bot._world(view)
    legal = world.legal_actions(seat)
    scored = []
    for action in view.actions:
        if action['type'] not in KINDS or action == served or action not in legal:
            continue
        child = deepcopy(world, {id(world.board): world.board})
        if action['type'] == 'buy_development':
            child.players[seat].pay('development')
            score = bot._leaf(child, seat) + bot.purchase_value
        elif action['type'] == 'end_turn':
            score = bot._leaf(child, seat)
        else:
            child.apply(seat, action)
            score = bot._leaf(child, seat)
        scored.append((score, len(scored), action))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [served] + [a for _, _, a in scored[:limit - 1]]


def playout(game, bots, seat_action, seat, stream, max_turns):
    from catan_ai.observation import Observation
    game = deepcopy(game)
    bots = deepcopy(bots)
    game.rng = random.Random(stream)
    game.deck = list(game.deck)
    game.rng.shuffle(game.deck)
    game.apply(seat, seat_action)
    while game.winner is None and game.turn < max_turns:
        s = game.current
        game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    return game.winner, game.winner is None, game


def run_game(args):
    seed, config, path, identity_sha = args
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
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
    pick = random.Random(seed * 7919 + 1)
    decisions, value_rows, seen, truncated = [], [], set(), 0
    while game.winner is None and game.turn < config['max_turns']:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        first = game.phase == 'actions' and (game.turn, seat) not in seen
        if first:
            seen.add((game.turn, seat))
        points = config.get('value_points', ['roll'])
        if 'roll' in points and game.phase == 'roll' and (game.turn, seat, 'roll') not in seen:
            seen.add((game.turn, seat, 'roll'))
            value_rows.append({'turn': game.turn, 'seat': seat, 'point': 'roll',
                               **features(game, seat, deepcopy(bots[seat]))})
        probe = deepcopy(bots[seat])
        action = bots[seat].choose(view, seat)
        if ('end_turn' in points and game.phase == 'actions' and action['type'] == 'end_turn'
                and (game.turn, seat, 'end') not in seen):
            seen.add((game.turn, seat, 'end'))
            value_rows.append({'turn': game.turn, 'seat': seat, 'point': 'end_turn',
                               **features(game, seat, deepcopy(probe))})
        if first and action['type'] in KINDS and pick.random() < config['decision_rate']:
            pre = features(game, seat, deepcopy(probe))
            rows = []
            for k, cand in enumerate(candidates(view, seat, probe, action, config['candidates'])):
                after = deepcopy(game)
                after_bot = deepcopy(probe)
                after.apply(seat, cand)
                post = features(after, seat, after_bot)
                row = {'action': cand, 'act': action_encoding(cand), 'served': k == 0,
                       'after': post['features'], 'leaf': post['leaf'], 'shared': post['shared'],
                       'wins': 0, 'playouts': 0, 'truncated': 0}
                for r in range(config['playouts']):
                    winner, trunc, _ = playout(game, [deepcopy(b) for b in bots[:seat]] + [deepcopy(probe)]
                                               + [deepcopy(b) for b in bots[seat + 1:]], cand, seat,
                                               seed * 1_000_003 + game.turn * 101 + r, config['max_turns'])
                    row['wins'] += winner == seat
                    row['playouts'] += 1
                    row['truncated'] += trunc
                    truncated += trunc
                rows.append(row)
            decisions.append({'turn': game.turn, 'seat': seat, 'points': game.players[seat].points,
                              'obs': pre['features'], 'candidates': rows})
        game.apply(seat, action)
    for row in value_rows:
        row['won'] = game.winner == row['seat']
    record = {'seed': seed, 'identity': identity_sha, 'schema': 2, 'value_rows': value_rows,
              'winner': game.winner, 'turns': game.turn,
              'truncated_game': game.winner is None, 'truncated_playouts': truncated,
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
    lo, n = config['seeds']
    jobs = [(seed, config, str(directory / f'{seed}.json'), identity['identity_sha256'])
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
