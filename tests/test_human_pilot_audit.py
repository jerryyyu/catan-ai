import copy
from pathlib import Path
import runpy

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/audit_human_pilot.py'))['audit']


def fixture():
    seed = 1480100
    names = ['Bot', 'PRIVATE NAME A', 'PRIVATE NAME B', 'PRIVATE NAME C']
    game = Game.new(names, seed)
    bot = make_policy('tactical', seed * 31)
    trace = []
    while game.current == 0:
        action = bot.choose(Observation.for_policy(game, 0, bot), 0)
        game.apply(0, action)
        trace.append({'seat': 0, 'actor': 'bot', 'action': action, 'seconds': .01})
    return {'schema': 1, 'source': source_files(), 'seed': seed, 'names': names,
            'bot': 'tactical', 'bot_seat': 0, 'trace': trace,
            'participants': {str(s): {'name': names[s], 'completed_matches': 50, 'consent': True}
                             for s in (1, 2, 3)}, 'winner': game.winner, 'turn': game.turn}


def test_audit_validates_partial_replay_without_promoting_synthetic_human_labels():
    record = fixture()
    original = copy.deepcopy(record)
    result = audit(record)
    assert result['replay_valid'] and not result['completed']
    assert result['registered_experienced_seats'] == 3
    assert result['human_participation_verified'] is False
    assert 'PRIVATE NAME' not in str(result)
    assert record == original


@pytest.mark.parametrize('defect', ['source', 'consent', 'actor', 'timing', 'winner', 'action'])
def test_audit_rejects_inconsistent_record(defect):
    record = fixture()
    if defect == 'source':
        record['source'] = {}
    elif defect == 'consent':
        record['participants']['1']['consent'] = False
    elif defect == 'actor':
        record['trace'][0]['actor'] = 'human'
    elif defect == 'timing':
        record['trace'][0]['seconds'] = float('nan')
    elif defect == 'winner':
        record['winner'] = 1
    else:
        record['trace'][0]['action'] = {'type': 'roll'}
    with pytest.raises(ValueError):
        audit(record)


def test_room_audit_aggregates_recorded_turn_cost_and_excludes_setup():
    record = fixture()
    game = Game.new(record['names'], record['seed'])
    bots = [make_policy('tactical', record['seed'] * 31 + seat) for seat in range(4)]
    trace = []
    totals = {}
    # Include setup and multiple ordinary turns, without requiring a full game.
    while game.turn < 5:
        seat = game.current
        action = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
        setup = game.phase.startswith('setup')
        seconds = 50.0 if setup else 6.0
        if seat == 0 and not setup:
            totals[game.turn] = totals.get(game.turn, 0) + seconds
        trace.append({'seat': seat, 'actor': 'bot' if seat == 0 else 'human',
                      'action': action, 'seconds': seconds})
        game.apply(seat, action)
    record.update(trace=trace, winner=game.winner, turn=game.turn)
    original = copy.deepcopy(record)
    result = audit(record)
    assert len(totals) >= 2
    assert result['maximum_recorded_bot_turn_seconds'] == max(totals.values())
    assert result['recorded_turn_budget_violations'] == sum(t > 10 for t in totals.values())
    assert result['recorded_step_budget_violations'] == sum(row['seat'] == 0 for row in trace)
    assert result['maximum_recorded_bot_step_seconds'] == 50.0
    assert record == original


def test_setup_only_has_no_recorded_turn_budget():
    result = audit(fixture())
    assert result['maximum_recorded_bot_turn_seconds'] is None
    assert result['recorded_turn_budget_violations'] == 0
    assert result['recorded_step_budget_violations'] == 0
