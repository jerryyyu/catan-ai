from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.observation import Observation


@pytest.mark.parametrize('name,opening,turn', [
    ('planner_tactical_opening', 'tactical', 'planner_robber'),
    ('tactical_planner_opening', 'planner_robber', 'tactical')])
def test_component_actions_match_selected_policy(name, opening, turn):
    game = Game.new(['a', 'b', 'c', 'd'], 1472900)
    bot = make_policy(name, 1)
    view = Observation.for_policy(game, 0, bot)
    assert bot.choose(view, 0) == make_policy(opening, 1).choose(view, 0)
    game.phase, game.turn = 'actions', 8
    view = Observation.for_policy(game, 0, bot)
    reference = make_policy(turn, 1)
    assert bot.choose(view, 0) == reference.choose(view, 0)
    assert bot.turn.rng.getstate() == reference.rng.getstate()


def test_component_reader_requires_both_panels_and_complete_evidence():
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/probe_policy_components.py'))
    rows = []
    identity = {'commit': 'ecf931181b9a65bb4116a2153fb78c16f1438e00'}
    for panel, start in [('tactical', 1473000), ('mixed', 1473100)]:
        for seed in range(start, start + 8):
            for seat in range(4):
                games = {}
                for arm in module['ARMS']:
                    games[arm] = {'winner': seat if arm == 'planner_tactical_opening' else (seat + 1) % 4,
                                  'truncated': False, 'step_seconds': [[.1] * 4] * 4,
                                  'turn_timings': [{'seconds': .2, 'complete': True}], 'external': {}}
                    if 'external_opening' in arm:
                        games[arm]['external'][str(seat)] = {'identity': identity, 'diagnostics': [
                            {'phase': p, 'fallback': False, 'budget_failure': False}
                            for p in ['setup_settlement', 'setup_road'] * 2]}
                rows.append({'panel': panel, 'seed': seed, 'seat': seat, 'games': games})
    result = module['summarize'](rows)
    assert result['selected_for_fresh_dev'] == 'planner_tactical_opening'
    assert result['pass_for_promotion'] is False
    failed = deepcopy(rows)
    failed[0]['games']['planner_tactical_opening']['step_seconds'][0][0] = 3.
    assert module['summarize'](failed)['selected_for_fresh_dev'] is None
    failed = deepcopy(rows)
    failed[0] = deepcopy(failed[1])
    with pytest.raises(ValueError, match='incomplete'):
        module['summarize'](failed)
