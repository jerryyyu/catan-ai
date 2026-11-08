"""Build the Catan AI JSettlers bridge and verify every external binary it runs with.

``build()`` compiles ``tools/jsettlers/java`` (clean output directory) against the JSettlers2
classes into ``runs/external/jsettlers-build/catanai-classes``.

``verify_jsettlers()`` fails closed unless the loaded JSettlers artifacts are exactly the pinned
release: the checkout is at the pinned commit with no tracked or untracked changes under ``src/``,
a fresh ``javac`` of the pinned sources with the pinned JDK reproduces every loaded ``.class``
byte-for-byte (no missing or extra classes), every other file on the class directory equals the
checkout's ``src/main/resources`` copy, and the Gson jar has its pinned SHA-256.

``build_manifest()`` does both and returns the identity of everything on the run's class path
(per-file hashes plus aggregate digests, JDK version, our Java sources and classes). It is written
to ``runs/external/jsettlers-build/catanai-build-manifest.json`` and sealed into each run.
Never vendors JSettlers source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'runs' / 'external'
BUILD = EXTERNAL / 'jsettlers-build'
CHECKOUT = EXTERNAL / 'jsettlers2'
JAVA_HOME = EXTERNAL / 'jdk' / 'jdk-17.0.21+9' / 'Contents' / 'Home'
SOURCES = ROOT / 'tools' / 'jsettlers' / 'java'
OUT = BUILD / 'catanai-classes'
MANIFEST = BUILD / 'catanai-build-manifest.json'

JSETTLERS_TAG = 'release-2.6.10'
JSETTLERS_COMMIT = 'c5141056d8e3fff548817b3a1712eb66085e897b'
GSON_JAR = 'gson-2.8.6.jar'
GSON_SHA256 = 'c8fb4839054d280b3033f800d1f5a97de2f028eb8ba2eb458ad287e536f3f25f'
JDK_VERSION = '17.0.21+9'


class BuildIdentityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path, pattern: str = '*') -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob(pattern)) if p.is_file()}


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def classpath(include_bridge: bool = True, build_dir: Path = BUILD) -> str:
    parts = [build_dir / 'classes', build_dir / GSON_JAR]
    if include_bridge:
        parts.append(build_dir / 'catanai-classes')
    return ':'.join(str(p) for p in parts)


def java(tool: str = 'java', java_home: Path = JAVA_HOME) -> str:
    return str(java_home / 'bin' / tool)


def build(out: Path = OUT) -> Path:
    if out.exists():
        shutil.rmtree(out)  # no stale classes from deleted or renamed sources
    out.mkdir(parents=True)
    files = sorted(str(p) for p in SOURCES.rglob('*.java'))
    if not files:
        raise SystemExit('no Java sources under ' + str(SOURCES))
    cmd = [java('javac'), '-nowarn', '-encoding', 'UTF-8', '--release', '17',
           '-cp', classpath(False), '-d', str(out), *files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    return out


def check_checkout(checkout: Path = CHECKOUT, pin: str = JSETTLERS_COMMIT) -> str:
    """The checkout must be at ``pin`` with no tracked or untracked change under src/."""
    try:
        head = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildIdentityError(f'cannot read JSettlers checkout {checkout}: {exc}') from exc
    if head != pin:
        raise BuildIdentityError(f'JSettlers checkout is at {head}, pinned {pin}')
    status = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=all',
                             '--', 'src'], capture_output=True, text=True).stdout.strip()
    if status:
        raise BuildIdentityError('JSettlers checkout has changes under src/: ' + status.splitlines()[0])
    return head


def compare_trees(expected: dict[str, str], loaded: dict[str, str], what: str) -> None:
    missing = sorted(set(expected) - set(loaded))
    extra = sorted(set(loaded) - set(expected))
    changed = sorted(k for k in set(expected) & set(loaded) if expected[k] != loaded[k])
    if missing or extra or changed:
        raise BuildIdentityError(f'{what} do not match the pinned build: missing {missing[:3]}, '
                                 f'extra {extra[:3]}, changed {changed[:3]}')


def verify_jsettlers(build_dir: Path = BUILD, checkout: Path = CHECKOUT, java_home: Path = JAVA_HOME,
                     pin: str = JSETTLERS_COMMIT, gson_sha256: str = GSON_SHA256) -> dict:
    """Fail closed unless the loaded JSettlers classes and jar are the pinned release."""
    commit = check_checkout(checkout, pin)
    jar = build_dir / GSON_JAR
    if not jar.is_file() or sha256(jar) != gson_sha256:
        raise BuildIdentityError(f'{jar} is missing or not the pinned Gson jar')
    classes = build_dir / 'classes'
    loaded = tree_hashes(classes)
    loaded_classes = {k: v for k, v in loaded.items() if k.endswith('.class')}
    loaded_resources = {k: v for k, v in loaded.items() if not k.endswith('.class')}
    sources = sorted(str(p) for p in (checkout / 'src' / 'main' / 'java').rglob('*.java'))
    with tempfile.TemporaryDirectory(prefix='jsettlers-rebuild-') as tmp:
        result = subprocess.run([java('javac', java_home), '-nowarn', '-encoding', 'UTF-8', '-cp', str(jar),
                                 '-d', tmp, *sources], capture_output=True, text=True)
        if result.returncode != 0:
            raise BuildIdentityError('rebuilding pinned JSettlers failed: ' + result.stderr[-500:])
        rebuilt = tree_hashes(Path(tmp), '*.class')
    compare_trees(rebuilt, loaded_classes, 'loaded JSettlers classes')
    resources = checkout / 'src' / 'main' / 'resources'
    compare_trees({k: sha256(resources / k) for k in loaded_resources if (resources / k).is_file()}
                  | {k: '' for k in loaded_resources if not (resources / k).is_file()},
                  loaded_resources, 'loaded JSettlers resources')
    return {'tag': JSETTLERS_TAG, 'commit': commit, 'classes_digest': digest(loaded),
            'class_files': len(loaded_classes), 'resource_files': len(loaded_resources),
            'gson_jar': GSON_JAR, 'gson_sha256': gson_sha256, 'files': loaded}


def java_version(java_home: Path = JAVA_HOME) -> str:
    out = subprocess.run([java('java', java_home), '-version'], capture_output=True, text=True)
    return (out.stderr or out.stdout).strip()


def build_manifest(write: bool = True) -> dict:
    """Build our classes, verify JSettlers, and return the identity of the whole class path."""
    jsettlers = verify_jsettlers()
    version = java_version()
    if JDK_VERSION not in version:
        raise BuildIdentityError(f'JDK is not {JDK_VERSION}: {version.splitlines()[0]}')
    build()
    ours_src = tree_hashes(SOURCES, '*.java')
    ours_cls = tree_hashes(OUT)
    manifest = {
        'schema': 1,
        'jdk': {'home': str(JAVA_HOME), 'version': version},
        'jsettlers': jsettlers,
        'bridge_java': {'sources_digest': digest(ours_src), 'sources': ours_src,
                        'classes_digest': digest(ours_cls), 'classes': ours_cls},
        'classpath': classpath(),
    }
    if write:
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def verify_loaded(manifest: dict, build_dir: Path = BUILD) -> None:
    """Re-hash the class path right before launch; any change since the manifest fails closed."""
    compare_trees(manifest['jsettlers']['files'], tree_hashes(build_dir / 'classes'), 'JSettlers classes')
    if sha256(build_dir / GSON_JAR) != manifest['jsettlers']['gson_sha256']:
        raise BuildIdentityError('Gson jar changed since the build manifest')
    compare_trees(manifest['bridge_java']['classes'], tree_hashes(build_dir / 'catanai-classes'),
                  'bridge classes')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-boards', type=int, default=0,
                        help='after building, print N JSettlers-generated boards as JSON')
    parser.add_argument('--verify', action='store_true',
                        help='also verify the JSettlers artifacts and write the build manifest')
    args = parser.parse_args()
    if args.verify:
        manifest = build_manifest()
        print(f'verified JSettlers {manifest["jsettlers"]["commit"]} '
              f'({manifest["jsettlers"]["class_files"]} classes); manifest {MANIFEST}', file=sys.stderr)
    else:
        print(f'built {build()}', file=sys.stderr)
    if args.dump_boards:
        subprocess.run([java(), '-cp', classpath(), 'catanai.jsettlers.BoardDump', str(args.dump_boards)],
                       check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
