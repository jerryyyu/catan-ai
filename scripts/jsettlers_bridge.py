"""Launch the Python side of the JSettlers bridge (started by the Java robot client)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'jsettlers'))

from jsbridge.server import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
