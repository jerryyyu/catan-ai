"""Compile the Catan AI JSettlers bridge (tools/jsettlers/java) against JSettlers2 classes.

Output: runs/external/jsettlers-build/catanai-classes. Never vendors JSettlers source;
our classes only compile against the locally built JSettlers class directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'runs' / 'external'
BUILD = EXTERNAL / 'jsettlers-build'
JAVA_HOME = EXTERNAL / 'jdk' / 'jdk-17.0.21+9' / 'Contents' / 'Home'
SOURCES = ROOT / 'tools' / 'jsettlers' / 'java'
OUT = BUILD / 'catanai-classes'


def classpath(include_bridge: bool = True) -> str:
    parts = [BUILD / 'classes', BUILD / 'gson-2.8.6.jar']
    if include_bridge:
        parts.append(OUT)
    return ':'.join(str(p) for p in parts)


def java(tool: str = 'java') -> str:
    return str(JAVA_HOME / 'bin' / tool)


def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in SOURCES.rglob('*.java'))
    if not files:
        raise SystemExit('no Java sources under ' + str(SOURCES))
    cmd = [java('javac'), '-nowarn', '-encoding', 'UTF-8', '--release', '17',
           '-cp', classpath(False), '-d', str(OUT), *files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    return OUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-boards', type=int, default=0,
                        help='after building, print N JSettlers-generated boards as JSON')
    args = parser.parse_args()
    out = build()
    print(f'built {out}', file=sys.stderr)
    if args.dump_boards:
        subprocess.run([java(), '-cp', classpath(), 'catanai.jsettlers.BoardDump', str(args.dump_boards)],
                       check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
