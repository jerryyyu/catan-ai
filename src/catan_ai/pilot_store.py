"""Private local pilot records, atomically replaced after each API action batch."""
import json
import os
from pathlib import Path
import re
import tempfile


def record_path(game_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', game_id):
        raise FileNotFoundError('invalid game identifier')
    return Path(os.environ.get('CATAN_PILOT_DIR', 'runs/pilots')) / (game_id + '.json')


def save(game_id, record):
    path = record_path(game_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load(game_id):
    return json.loads(record_path(game_id).read_text())
