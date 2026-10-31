from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT, load_parameters
from catan_ai.search_eval import model_identity


def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    data = {'schema': 1, 'weights': DEFAULT, 'source': source_files()}
    path.write_text(json.dumps(data))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))
    return path, data


def test_default_weights_preserve_full_planner_trajectory(tmp_path, monkeypatch):
    checkpoint(tmp_path, monkeypatch)
    reference, candidate = [make_policy(name, 17) for name in ('planner_robber', 'planner_parameter')]
    game = Game.new(['a', 'b', 'c', 'd'], 1476900)
    opponents = [make_policy('tactical', 17 + s) for s in range(4)]
    for _ in range(5000):
        if game.winner is not None:
            break
        seat = game.current
        view = Observation.from_game(game, seat, include_events=False)
        if seat == 0:
            before = deepcopy(view)
            action = candidate.choose(view, seat)
            assert action == reference.choose(view, seat)
            assert view == before and candidate.rng.getstate() == reference.rng.getstate()
        else:
            action = opponents[seat].choose(view, seat)
        game.apply(seat, action)
    assert game.winner is not None
    identity = model_identity({'candidate': 'planner_parameter', 'panels': []})
    assert identity['sha256'] == candidate.model_hash


@pytest.mark.parametrize('defect', ['nan', 'boolean', 'length', 'bounds', 'source'])
def test_invalid_or_incompatible_parameters_fail_closed(tmp_path, monkeypatch, defect):
    path, data = checkpoint(tmp_path, monkeypatch)
    data['weights'] = list(DEFAULT)
    if defect == 'nan':
        data['weights'][0] = float('nan')
    elif defect == 'boolean':
        data['weights'][2] = True
    elif defect == 'length':
        data['weights'].pop()
    elif defect == 'bounds':
        data['weights'][0] = 100
    else:
        data['source'] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_parameters()


def test_nondefault_parameters_preserve_hidden_card_boundary(tmp_path, monkeypatch):
    path, data = checkpoint(tmp_path, monkeypatch)
    data['weights'] = [17., .65, .64, 1.55, .44, 1.45, 7.06, 2.8]
    path.write_text(json.dumps(data))
    game = Game.new(['a', 'b', 'c', 'd'], 1476901)
    game.phase, game.turn = 'actions', 8
    game.players[0].settlements = {0}
    game.players[0].resources.update({Resource.ORE: 3, Resource.GRAIN: 2, Resource.BRICK: 4})
    game.players[1].resources[Resource.WOOL] = 2
    game.players[1].development = [('knight', 0)]
    changed = deepcopy(game)
    changed.players[1].resources[Resource.WOOL] = 0
    changed.players[1].resources[Resource.ORE] = 2
    changed.players[1].development = [('victory_point', 0)]
    changed.deck.reverse()
    view = Observation.from_game(game, 0, include_events=False)
    other = Observation.from_game(changed, 0, include_events=False)
    assert view == other
    before = deepcopy(view)
    first, second = [make_policy('planner_parameter', 17) for _ in range(2)]
    action = first.choose(view, 0)
    assert action == second.choose(other, 0) and action in view.actions
    assert first.rng.getstate() == second.rng.getstate() and view == before


def test_training_reader_requires_complete_seats_identity_and_clocks():
    summarize = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  'scripts/train_economic_parameters.py'))['summarize']
    rows = []
    for panel, start in [('tactical', 1477000), ('mixed', 1477100)]:
        for seed in range(start, start + 8):
            for seat in range(4):
                rows.append({'panel': panel, 'seed': seed, 'seat': seat, 'game': {
                    'winner': seat, 'truncated': False, 'external': {},
                    'policy_metadata': {str(seat): {'model_sha256': 'test', 'load_error': None}},
                    'step_seconds': [[.1]] * 4, 'turn_timings': [{'complete': True, 'seconds': .2}]}})
    assert summarize(rows, 'test')['fitness'] == [1, 1]
    changed = deepcopy(rows)
    changed[0]['game']['turn_timings'][0]['seconds'] = 11
    assert summarize(changed, 'test')['valid'] is False
    changed[0] = deepcopy(changed[1])
    with pytest.raises(ValueError, match='duplicate'):
        summarize(changed, 'test')
    with pytest.raises(ValueError, match='identity'):
        summarize(rows, 'different')
