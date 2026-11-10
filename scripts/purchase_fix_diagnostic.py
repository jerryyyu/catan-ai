"""How often planner_pv2_tiles_buyfix picks a different first move than planner_pv2_tiles (seat 0 vs three
planner_offer_accept), counting PV decisions and purchase involvement.

    uv run --with numpy python scripts/purchase_fix_diagnostic.py START COUNT
"""
import collections
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from catan_ai.purchase_scoring import super_pick  # noqa: E402
from catan_ai.unified_planner import AcceptanceOfferPlanner  # noqa: E402
from catan_ai.pv_search import REPLACEABLE  # noqa: E402
start, n = int(sys.argv[1]), int(sys.argv[2])
c = collections.Counter()
for s in range(start, start + n):
    game = Game.new(['planner_pv2_tiles_buyfix'] + ['planner_offer_accept'] * 3, s)
    bots = [make_policy('planner_pv2_tiles_buyfix', s)] + [make_policy('planner_offer_accept', s + i) for i in (1, 2, 3)]
    while game.winner is None and game.turn < 1000:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if seat == 0 and view.phase == 'actions' and any(a['type'] == 'end_turn' for a in view.actions):
            b = bots[0]
            before = b.last_offer_turn
            served = AcceptanceOfferPlanner.choose(b, view, seat)
            b.last_offer_turn = before
            if served['type'] in REPLACEABLE and served['type'] not in ('offer_trade', 'play_development'):
                fix, old = b._pick(view, seat), super_pick(b, view, seat)
                c['pv_decisions'] += 1
                buy_cand = any(a['type'] == 'buy_development' for a in view.legal_actions(seat))
                c['buy_legal'] += buy_cand
                if (fix or [None])[:1] != (old or [None])[:1]:
                    c['first_move_differs'] += 1
                    c['diff_old_buys'] += bool(old) and old[0]['type'] == 'buy_development'
                    c['diff_fix_buys'] += bool(fix) and fix[0]['type'] == 'buy_development'
        game.apply(seat, bots[seat].choose(view, seat))
    c['games'] += 1
print(json.dumps(dict(c)))
