"""Action-trace hashes of seeded games, for speed-only and cross-engine parity checks.

Each game puts POLICY in seat 0 and ``planner_offer_accept`` in the other seats; the hash covers every
(seat, action) in order. Prints JSON {policy: {seed: [sha16, winner, turns], 'seconds': s}}.

    uv run --with numpy python scripts/trace_parity.py planner_offer_accept,planner_pv2_tiles 4100001,4100002
"""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402

policies = sys.argv[1].split(',')
seeds = [int(s) for s in sys.argv[2].split(',')]
out = {}
for pol in policies:
    out[pol] = {}
    t0 = time.time()
    for s in seeds:
        game = Game.new([pol] + ['planner_offer_accept'] * 3, s)
        bots = [make_policy(pol, s)] + [make_policy('planner_offer_accept', s + i) for i in (1, 2, 3)]
        h = hashlib.sha256()
        while game.winner is None and game.turn < 1000:
            seat = game.current
            a = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
            h.update(json.dumps([seat, a], sort_keys=True).encode())
            game.apply(seat, a)
        out[pol][s] = [h.hexdigest()[:16], game.winner, game.turn]
    out[pol]['seconds'] = round(time.time() - t0, 1)
print(json.dumps(out))
