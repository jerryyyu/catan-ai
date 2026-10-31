import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from catan_ai.api import app
from catan_ai.pilot_store import record_path


def test_api_records_are_under_test_directory(tmp_path):
    created = TestClient(app).post('/api/games', json={'seed': 1456000}).json()
    path = record_path(created['game_id'])
    assert path.is_relative_to(tmp_path)
    assert json.loads(path.read_text())['seed'] == 1456000


def test_browser_server_ignores_participant_storage_environment(tmp_path):
    participant_dir = tmp_path / 'participants'
    participant_dir.mkdir()
    sentinel = participant_dir / 'preserve.json'
    sentinel.write_text('{"participant": true}')
    program = '''
import json
from pathlib import Path
from fastapi.testclient import TestClient
import browser_server
from catan_ai.pilot_store import record_path
created = TestClient(browser_server.app).post('/api/games', json={'seed': 1456001}).json()
path = record_path(created['game_id'])
assert path.exists()
print(json.dumps({'directory': str(path.parent)}))
'''
    env = {**os.environ, 'CATAN_PILOT_DIR': str(participant_dir)}
    result = subprocess.run([sys.executable, '-c', program],
                            cwd=Path(__file__).parent, env=env,
                            capture_output=True, text=True, check=True)
    directory = Path(json.loads(result.stdout)['directory'])
    assert directory != participant_dir
    assert not directory.exists()
    assert list(participant_dir.iterdir()) == [sentinel]
    assert sentinel.read_text() == '{"participant": true}'
