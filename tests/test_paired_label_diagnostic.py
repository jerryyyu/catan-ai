from pathlib import Path
import runpy

import pytest


def row(differences):
    return {'choices': [
        {'action': {'type': 'end_turn'}, 'rollouts': [{'win': int(x < 0)} for x in differences]},
        {'action': {'type': 'build_road'}, 'rollouts': [{'win': int(x > 0)} for x in differences]}]}


def test_split_half_detects_reliable_and_disagreeing_targets(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    summarize = runpy.run_path(str(scripts / 'diagnose_paired_labels.py'))['summarize']
    good = summarize([row([1, 1, 1, 1]), row([-1, -1, -1, -1])])
    assert good['split_half_correlation'] == pytest.approx(1.)
    assert good['estimated_four_replicate_mean_noise'] == 0
    assert good['mean_squared_target'] == 1
    assert good['pass'] is True
    bad = summarize([row([1, 1, -1, -1]), row([-1, -1, 1, 1])])
    assert bad['split_half_correlation'] == pytest.approx(-1.)
    assert bad['estimated_four_replicate_mean_noise'] == pytest.approx(1 / 3)
    assert bad['zero_target_fraction'] == 1
    assert bad['pass'] is False
    tied = summarize([row([0, 0, 0, 0])])
    assert tied['split_half_correlation'] is None and tied['pass'] is False
