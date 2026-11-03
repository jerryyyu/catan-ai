import runpy
from pathlib import Path

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
