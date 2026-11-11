"""Setup-phase value data and the placement test set (experiments/placement-value.json; board #1, Jerry 2026-11-11).

The value transformer was trained only on rows from turn 1 on, so it has never scored a setup position, and
the served bot places its opening settlements with the fitted economic score. Three declared modes
(``config['point']``):

* ``setup``: replay each declared self-play record's setup phase (served bots seeded ``seed*17+i+1``) and, after
  every setup settlement, encode the placing seat's view (encoder v2, ``bot._world``). Label = the record's
  winner is that seat. The replay is verified against the record's first stored turn-start row (seat, turn and
  all legacy features), the same check ``reencode_v2.py`` applies. Truncated records give no rows.
* ``setupfork``: fresh seeded games of ``policy``. One of the 8 setup settlements (index drawn per game) is
  replaced by a uniformly drawn legal vertex; that afterstate is encoded from the placing seat's view and the game
  is played to the end by the same bots. Label = that seat won; forks reaching ``max_turns`` are dropped.
* ``placetest``: fresh seeded games of ``policy`` (all four seats). At every setup settlement decision the
  candidates are the served economic ranking's top ``top_k`` plus ``random_k`` other legal vertices (seeded);
  each is played out ``playouts`` times from that decision (dice and deck reshuffled per playout, common random
  numbers across candidates; fresh bots of ``policy`` seeded per playout). Rows: afterstate features from the chooser's view,
  wins, the served (economic) choice. The real game continues with the served choice. One job per
  (game, decision, candidate), so the work parallelizes.

``setup``/``setupfork`` write shards with ``reencode_v2.write_shard`` (verified by ``train_value_v2.shards``);
``placetest`` writes one JSON per job and a sealed ``DONE.json``. Every DONE seals ``producer_provenance``.
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
from artifact_guard import producer_provenance  # noqa: E402
from reencode_v2 import _row, write_shard  # noqa: E402

SETUP_DECISIONS = 8
DEV_DECK = ['knight'] * 14 + ['victory_point'] * 5 + ['monopoly', 'year_of_plenty', 'road_building'] * 2


def _bots(policy, seed):
    from catan_ai.bots import make_policy
    return [make_policy(policy, seed * 17 + i + 1) for i in range(4)]


def _encode_seat(game, seat, bot):
    from catan_ai.encoding import encode
    from catan_ai.observation import Observation
    view = Observation.for_policy(game, seat, bot)
    return encode(view, bot._world(view))


def served_ranking(bot, view, seat):
    """The served economic setup ranking (``EconomicPlanner.choose``'s setup branch), best first."""
    from catan_ai.tactical import public_projection
    root = public_projection(view)
    legal = view.legal_actions(seat)

    def placement(action):
        child = deepcopy(root, {id(root.board): root.board})
        child.players[seat].settlements.add(action['vertex'])
        return bot._value(child, seat)
    return sorted(legal, key=placement, reverse=True)


def setup_rows(seed, path, identity, policy, max_turns):
    """Mode ``setup``: (rows [(x, y, key)], truncated)."""
    from catan_ai.encoding import encode
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    from catan_ai.value_model import value_features
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if record['seed'] != seed or record['identity'] != identity:
        raise ValueError(f'{path} does not belong to the declared root')
    if record['winner'] is None:
        return [], True, hashlib.sha256(raw).hexdigest()
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = _bots(policy, seed)
    rows = []
    while game.phase in ('setup_settlement', 'setup_road'):
        seat = game.current
        action = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
        game.apply(seat, action)
        if action['type'] == 'build_settlement':
            rows.append((_encode_seat(game, seat, bots[seat]), int(record['winner'] == seat), (0, seat, 0.0, 0.0)))
    stored = [r for r in record['value_rows'] if r['point'] == 'roll']
    seat = game.current
    view = Observation.for_policy(game, seat, bots[seat])
    _row(0, set(), stored, game, seat, view, bots[seat], [], [], [], encode, value_features)  # raises on mismatch
    if len(rows) != SETUP_DECISIONS:
        raise ValueError(f'seed {seed}: {len(rows)} setup settlements')
    return rows, False, hashlib.sha256(raw).hexdigest()


def setupfork_row(seed, policy, max_turns):
    """Mode ``setupfork``: ((x, y, key) or None, truncated)."""
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    pick = random.Random(seed * 104729 + 17)
    target = pick.randrange(SETUP_DECISIONS)
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = _bots(policy, seed)
    k, row = 0, None
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if game.phase == 'setup_settlement' and k == target:
            legal = view.legal_actions(seat)
            action = legal[pick.randrange(len(legal))]
            game.apply(seat, action)
            row = (_encode_seat(game, seat, bots[seat]), seat, (0, seat, 0.0, 0.0))
            k += 1
            continue
        if game.phase == 'setup_settlement':
            k += 1
        game.apply(seat, bots[seat].choose(view, seat))
    if game.winner is None or row is None:
        return None, game.winner is None
    x, seat, key = row
    return (x, int(game.winner == seat), key), False


def _to_decision(seed, policy, index):
    """The real game (served choices) just before setup decision ``index``; (game, bots, seat)."""
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    game = Game.new([f'{policy}-{i}' for i in range(4)], seed)
    bots = _bots(policy, seed)
    k = 0
    while True:
        seat = game.current
        if game.phase == 'setup_settlement':
            if k == index:
                return game, bots, seat
            k += 1
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))


def placetest_candidates(seed, policy, index, top_k, random_k):
    from catan_ai.observation import Observation
    game, bots, seat = _to_decision(seed, policy, index)
    view = Observation.for_policy(game, seat, bots[seat])
    ranked = served_ranking(bots[seat], view, seat)
    chosen = bots[seat].choose(view, seat)
    if chosen != ranked[0]:
        raise ValueError(f'seed {seed} decision {index}: served choice is not the top of the served ranking')
    rest = ranked[top_k:]
    extra = random.Random(seed * 1009 + index).sample(rest, min(random_k, len(rest)))
    return [(c, ranked.index(c)) for c in ranked[:top_k] + extra], seat


def placetest_job(args):
    seed, policy, index, vertex, rank, playouts, max_turns = args
    game, bots, seat = _to_decision(seed, policy, index)
    action = {'type': 'build_settlement', 'vertex': vertex}
    after = deepcopy(game)
    after.apply(seat, action)
    x = _encode_seat(after, seat, bots[seat])
    wins = truncated = 0
    for r in range(playouts):
        g = deepcopy(game)
        g.rng = random.Random(seed * 1_000_003 + index * 10_007 + r)  # common random numbers across candidates
        g.deck = list(DEV_DECK)
        g.rng.shuffle(g.deck)
        from catan_ai.bots import make_policy
        pbots = [make_policy(policy, seed * 31 + r * 4 + i) for i in range(4)]  # fresh copies per playout
        g.apply(seat, action)
        while g.winner is None and g.turn < max_turns:
            from catan_ai.observation import Observation
            t = g.current
            g.apply(t, pbots[t].choose(Observation.for_policy(g, t, pbots[t]), t))
        wins += g.winner == seat
        truncated += g.winner is None
    return {'seed': seed, 'setup_index': index, 'seat': seat, 'vertex': vertex, 'served_rank': rank,
            'served': rank == 0, 'x': [round(float(v), 5) for v in x], 'wins': wins, 'playouts': playouts,
            'truncated_playouts': truncated}


def _shard_job(args):
    mode, seed, cfg, src = args
    if mode == 'setup':
        path, identity, policy = src
        rows, truncated, sha = setup_rows(seed, path, identity, policy, cfg['max_turns'])
    else:
        row, truncated = setupfork_row(seed, cfg['policy'], cfg['max_turns'])
        rows, sha = ([row] if row else []), hashlib.sha256(f'setupfork:{seed}'.encode()).hexdigest()
    return seed, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], sha, truncated


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    mode = cfg['point']
    if mode not in ('setup', 'setupfork', 'placetest'):
        raise ValueError(f'unknown mode {mode}')
    args.output.mkdir(parents=True, exist_ok=False)
    prov = producer_provenance()
    started, written, dropped = time.monotonic(), {}, []
    if mode == 'placetest':
        jobs = []
        for _split, (lo, n) in cfg['split'].items():
            for seed in range(lo, lo + n):
                for index in range(SETUP_DECISIONS):
                    cands, _ = placetest_candidates(seed, cfg['policy'], index, cfg['top_k'], cfg['random_k'])
                    jobs += [(seed, cfg['policy'], index, c['vertex'], rank, cfg['playouts'], cfg['max_turns'])
                             for c, rank in cands]
        (args.output / 'rows').mkdir()
        with Pool(args.workers) as pool:
            for done, row in enumerate(pool.imap_unordered(placetest_job, jobs, chunksize=1), 1):
                name = args.output / 'rows' / f"{row['seed']}-{row['setup_index']}-{row['vertex']}.json"
                with name.open('x') as f:
                    json.dump(row, f)
                if done % 100 == 0:
                    print(json.dumps({'jobs': done, 'of': len(jobs), 'seconds': round(time.monotonic() - started)}), flush=True)
        files = sorted((args.output / 'rows').glob('*.json'))
        if len(files) != len(jobs):
            raise ValueError(f'{len(files)} rows for {len(jobs)} jobs')
        written = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    else:
        by_seed = {}
        if mode == 'setup':
            for root in cfg['roots']:
                manifest = json.loads((args.runs / root['root'] / 'manifest.json').read_text())
                if manifest['identity_sha256'] != root['identity_sha256']:
                    raise ValueError(f"{root['root']} identity differs from the declaration")
                policy = manifest['config'].get('policy', 'planner_offer_accept')
                for path in (args.runs / root['root'] / 'games').glob('*.json'):
                    by_seed[int(path.stem)] = (str(path), root['identity_sha256'], policy)
        for split, (lo, n) in cfg['split'].items():
            seeds = list(range(lo, lo + n))
            if mode == 'setup' and any(s not in by_seed for s in seeds):
                raise ValueError(f'{split}: seeds missing from the declared roots')
            for start in range(0, n, cfg['shard_games']):
                chunk = seeds[start:start + cfg['shard_games']]
                jobs = [(mode, s, cfg, by_seed.get(s)) for s in chunk]
                with Pool(args.workers) as pool:
                    results = pool.map(_shard_job, jobs, chunksize=4)
                name = write_shard(args.output, split, start // cfg['shard_games'], chunk, results)
                sidecar = json.loads(name.with_suffix('.json').read_text())
                written[name.stem] = sidecar['npz_sha256']
                dropped += sidecar['truncated_games']
                print(json.dumps({'split': split, 'shard': name.name, 'rows': sidecar['rows'],
                                  'seconds': round(time.monotonic() - started)}), flush=True)
    if producer_provenance()['source_files'] != prov['source_files']:
        raise ValueError('source changed while producing; not sealing')
    with (args.output / 'DONE.json').open('x') as stream:
        json.dump({'config': cfg, 'shards': written, 'truncated_games': dropped,
                   'seconds': time.monotonic() - started, 'provenance': prov}, stream)


if __name__ == '__main__':
    main()
