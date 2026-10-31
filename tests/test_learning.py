import copy
import json

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.learning import LearnedBot, action_features, state_features
from catan_ai.learn_pipeline import read_split, train, evaluate
from catan_ai.observation import Observation
from catan_ai.search_eval import model_identity


def test_features_use_only_observation_and_model_masks_actions(tmp_path):
    game = Game.new(['a', 'b', 'c', 'd'], 1200030)
    game.phase = 'actions'
    game.players[0].settlements = {0}
    game.players[0].resources[Resource.ORE] = 3
    game.players[0].resources[Resource.GRAIN] = 2
    hidden = copy.deepcopy(game)
    hidden.players[1].resources[Resource.ORE] = 15
    hidden.deck.reverse()
    a, b = Observation.from_game(game, 0), Observation.from_game(hidden, 0)
    assert state_features(a) == state_features(b)
    for action in a.actions:
        assert action_features(a, action) == action_features(b, action)
        assert len(action_features(a, action)) == 26
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'policy_weights': [0.] * 26, 'value_weights': [0.] * 21}))
    model = LearnedBot(1, path)
    assert model.choose(a, 0) in a.actions
    assert model.win_probability(a) == .5


def test_missing_or_nonfinite_model_falls_back_to_tactical(tmp_path):
    view = Observation.from_game(Game.new(['a', 'b', 'c', 'd'], 1200031), 0)
    missing = LearnedBot(1, tmp_path / 'missing')
    assert missing.choose(view, 0) == make_policy('tactical', 1).choose(view, 0)
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'schema': 1, 'policy_weights': [float('nan')] * 26, 'value_weights': [0.] * 21}))
    assert LearnedBot(1, path).model is None


def test_strength_screen_refuses_missing_model_and_pins_exact_bytes(tmp_path, monkeypatch):
    config = {'candidate': 'learned', 'panels': []}
    monkeypatch.delenv('CATAN_MODEL_PATH', raising=False)
    with pytest.raises(ValueError):
        model_identity(config)
    path = tmp_path / 'model.json'
    model = {'schema': 1, 'policy_weights': [0.] * 26, 'value_weights': [0.] * 21}
    path.write_text(json.dumps(model))
    monkeypatch.setenv('CATAN_MODEL_PATH', str(path))
    before = model_identity(config)
    model['policy_weights'][0] = 1
    path.write_text(json.dumps(model))
    assert model_identity(config)['sha256'] != before['sha256']


def test_split_boundary_and_actual_training_checkpoint(tmp_path):
    row = {'state': [1.] + [0.] * 20, 'features': [[1.] + [0.] * 25, [0.] * 26],
           'teacher': [{'wins': 2, 'completed': 2}, {'wins': 0, 'completed': 2}], 'win': 1}
    for split, seed in [('train', 1230000), ('validation', 1231000)]:
        folder = tmp_path / split
        folder.mkdir()
        (folder / f'{seed}.json').write_text(json.dumps({
            'identity': {'split': split, 'seed': seed, 'source': {'test': 'fixture'}},
            'complete': True, 'rows': [row]}))
    model = tmp_path / 'model.json'
    train(tmp_path, model, 40, 1)
    result = tmp_path / 'metrics.json'
    evaluate(tmp_path, 'validation', model, result)
    metrics = json.loads(result.read_text())
    assert metrics['policy_cross_entropy'] < metrics['uniform_cross_entropy']
    assert metrics['value_brier'] < metrics['constant_quarter_brier']
    invalid = tmp_path / 'validation' / '1231000.json'
    shard = json.loads(invalid.read_text())
    shard['identity']['seed'] = 1230000
    invalid.write_text(json.dumps(shard))
    with pytest.raises(ValueError):
        read_split(tmp_path, 'validation')
