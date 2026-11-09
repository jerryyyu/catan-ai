"""Launch the Python side of the StacSettlers bridge (started by the Java STAC robot brain)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'jsettlers'))
sys.path.insert(0, str(ROOT / 'tools' / 'stac'))

from stacbridge.server import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
