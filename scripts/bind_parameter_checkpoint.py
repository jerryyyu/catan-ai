"""Rebind the served fitted-planner checkpoint to the current source.

The checkpoint pins the hash of every source file, so any change under
``src/catan_ai`` makes it refuse to load. This script is the only supported way
to move it forward. It keeps the weights, records the new source hashes, and
then replays the checkpoint's compatibility games (seeded, all four seats
against Tactical). The binding is written only if every game reproduces the
recorded traces, outcomes and final engine RNG states exactly, so a rebind can
never silently change how the promoted bot plays.

Usage:
    uv run python scripts/bind_parameter_checkpoint.py            # check only
    uv run python scripts/bind_parameter_checkpoint.py --write    # rebind
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from catan_ai.experiments import source_files
from catan_ai.parameter_planner import SERVED_CHECKPOINT, compatibility_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=SERVED_CHECKPOINT)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.checkpoint.read_text())
    source = source_files()
    if data['source'] == source:
        print('checkpoint already bound to the current source')
    candidate = dict(data, source=source)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
        json.dump(candidate, handle, indent=1)
        handle.write('\n')
        temporary = handle.name
    os.environ['CATAN_PARAMETER_MODEL'] = temporary
    try:
        digest = compatibility_digest(candidate['compatibility']['plan'])
    finally:
        os.unlink(temporary)
    expected = data['compatibility']['trace_sha256']
    if digest != expected:
        sys.exit(f'compatibility traces changed: {digest} != {expected}; this source plays differently, '
                 'so it needs a new evaluation, not a rebind')
    print(f'compatibility traces identical ({digest[:12]})')
    if args.write and data['source'] != source:
        args.checkpoint.write_text(json.dumps(candidate, indent=1) + '\n')
        print(f'rebound {args.checkpoint}')
    elif data['source'] != source:
        sys.exit('checkpoint is bound to a different source; rerun with --write')


if __name__ == '__main__':
    main()
