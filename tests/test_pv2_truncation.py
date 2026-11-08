"""PV2 rollouts that hit max_turns are censored, counted, and gated (Codex HOLD @c32a17a, P2)."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pv2_data  # noqa: E402
import train_pv2  # noqa: E402

CONFIG = {'policy': 'planner_offer_accept', 'max_turns': 14, 'decision_rate': 1.0, 'candidates': 2, 'explore': 0.0}


def _game_at_actions():
    from catan_ai.bots import make_policy
    from catan_ai.engine import Game
    from catan_ai.observation import Observation
    game = Game.new([f'p-{i}' for i in range(4)], 5)
    bots = [make_policy('planner_offer_accept', 5 * 17 + i + 1) for i in range(4)]
    while not (game.phase == 'actions' and any(a['type'] == 'end_turn' for a in game.legal_actions(game.current))):
        s = game.current
        game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    return game, bots


def test_playout_reports_truncation_separately_from_loss_and_illegal():
    game, bots = _game_at_actions()
    seat = game.current
    assert pv2_data.playout(game, bots, seat, [], 1, game.turn + 1) == 'truncated'
    assert pv2_data.playout(game, bots, seat, [{'type': 'build_city', 'vertex': -1}], 1, 1000) == 'illegal'
    assert pv2_data.playout(game, bots, seat, [], 1, 1000) in ('win', 'loss')


def test_forced_small_max_turns_records_truncations_not_losses(tmp_path):
    path = tmp_path / '3.json'
    assert pv2_data.run_game((3, CONFIG, str(path), 'id', 2)) == 'done'
    record = json.loads(path.read_text())
    cands = [c for d in record['decisions'] for c in d['candidates']]
    assert cands and all('truncated' in c for c in cands)
    # every rollout from a game this short is censored at max_turns: none may count as a played loss
    assert record['rollout_truncations'] == sum(c['truncated'] for c in cands) > 0
    for c in cands:
        assert c['playouts'] + c['truncated'] + c['illegal'] == 2
        assert c['wins'] == c['playouts'] == 0
    assert train_pv2.truncation([record])['violation']


def _record(truncated):
    cand = {'seq': [], 'served': True, 'won_now': False, 'wins': 1, 'playouts': 2, 'illegal': 0}
    if truncated is not None:
        cand['truncated'] = truncated
    return {'decisions': [{'candidates': [cand, dict(cand, served=False)]}]}


@pytest.mark.parametrize('truncated, ok', [(0, True), (1, False), (None, False)])
def test_gate_truncation_rule(truncated, ok):
    out = train_pv2.truncation([_record(truncated)])
    assert (out['violation'] is None) == ok
    assert out['rollout_truncations'] == (None if truncated is None else 2 * truncated)
