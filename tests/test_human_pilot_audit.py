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
