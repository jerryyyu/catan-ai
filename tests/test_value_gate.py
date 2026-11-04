import json
import runpy
from pathlib import Path

import pytest

from catan_ai.experiments import source_files
from catan_ai.parameter_planner import DEFAULT

gate = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/value_gate.py'))


def test_auc_handles_ties_and_weights():
    auc = gate['auc']
    assert auc([1, 2, 3, 4], [0, 0, 1, 1], [1] * 4) == 1
    assert auc([4, 3, 2, 1], [0, 0, 1, 1], [1] * 4) == 0
    assert auc([1, 1], [0, 1], [1, 1]) == .5
    assert auc([1, 2, 3], [0, 1, 0], [1, 1, 2]) == 1 / 3
    assert auc([1, 2], [1, 1], [1, 1]) is None


def test_fit_separates_a_clear_signal():
    games = [{'rows': [{'features': [x, 0.], 'won': x > 0} for x in (-2., -1., 1., 2.)], 'truncated': False}]
    model = gate['fit'](games * 5, {'epochs': 30, 'learning_rate': .5, 'l2': 0., 'sgd_seed': 1})
    assert gate['predict'](model, [2., 0.]) > .5 > gate['predict'](model, [-2., 0.])


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _config(**changes):
    config = {'policy': 'planner_offer_accept', 'max_turns': 3, 'sample': 'end_turn',
              'splits': {'train': [10, 2], 'test': [20, 1]},
              'model': {'epochs': 1, 'learning_rate': .05, 'l2': 0., 'sgd_seed': 7},
              'bootstrap': {'resamples': 5, 'seed': 1},
              'gate': {'maximum_truncated_games': 9, 'minimum_auc_gain_lcb': 0.}}
    config.update(changes)
    return config


def test_resume_rejects_a_changed_identity_and_wrong_records(tmp_path, checkpoint):
    root = tmp_path / 'run'
    identity = gate['bind_manifest'](root, gate['run_identity'](_config()))
    path = root / 'train' / '10.json'
    path.parent.mkdir(parents=True)
    job = ('planner_offer_accept', 10, 3, str(path), 'end_turn', identity['identity_sha256'])
    assert gate['play'](job) == 'done' and gate['play'](job) == 'exists'
    with pytest.raises(ValueError):  # a different sample mode is a different run
        gate['bind_manifest'](root, gate['run_identity'](_config(sample='turn_start')))
    with pytest.raises(ValueError):  # a record under another seed's name
        gate['play'](('planner_offer_accept', 11, 3, str(path), 'end_turn', identity['identity_sha256']))
    with pytest.raises(ValueError):  # a record from another run identity
        gate['play'](('planner_offer_accept', 10, 3, str(path), 'end_turn', 'other'))


def test_identity_rejects_overlapping_splits(checkpoint):
    with pytest.raises(ValueError):
        gate['run_identity'](_config(splits={'train': [10, 5], 'test': [12, 2]}))


def test_evaluate_binds_the_corpus_and_refuses_mixed_records(tmp_path, checkpoint):
    root = tmp_path / 'run'
    config = _config()
    identity = gate['bind_manifest'](root, gate['run_identity'](config))
    for split, (lo, n) in config['splits'].items():
        (root / split).mkdir(parents=True)
        for seed in range(lo, lo + n):
            gate['play'](('planner_offer_accept', seed, 3, str(root / split / f'{seed}.json'),
                          'end_turn', identity['identity_sha256']))
    record = json.loads((root / 'test' / '20.json').read_text())
    record['identity'] = 'other'
    (root / 'test' / '20.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        gate['evaluate'](root, config, tmp_path / 'out.json')
