import json
import runpy
from pathlib import Path

import pytest

from catan_ai.experiments import source_files
from catan_ai.parameter_planner import DEFAULT

rank = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/rank_data.py'))


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def test_manifest_and_records_fail_closed(tmp_path):
    config = {'policy': 'planner_offer_accept', 'max_turns': 1000, 'seeds': [10, 1], 'decision_rate': 0.,
              'candidates': 2, 'playouts': 1}
    root = tmp_path / 'run'
    identity = rank['bind_manifest'](root, rank['run_identity'](config))
    with pytest.raises(ValueError):
        rank['bind_manifest'](root, rank['run_identity'](dict(config, playouts=2)))
    path = root / 'games' / '10.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seed': 11, 'identity': identity['identity_sha256']}))
    with pytest.raises(ValueError):
        rank['run_game']((10, config, str(path), identity['identity_sha256']))
