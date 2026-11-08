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

Builds are **private**: ``build_manifest(build_dir)`` stages everything a run loads (JSettlers
classes, the Gson jar, our bridge classes) into a new directory that must not exist yet, normally
inside the run's own output directory. Nothing shared is ever deleted or rewritten, so a second run,
a failed build or a verification command cannot change a class path a live JVM is reading. The
unpatched release classes are copied (read only) from ``runs/external/jsettlers-build/classes``.

A run may declare a source patch (``{"file": ..., "sha256": ...}``, e.g. the opt-in seeding patch
``runs/external/jsettlers-seed.patch``). Then the patch file must have exactly the declared
SHA-256, the checkout must still be the clean pinned commit, the pinned ``src/`` tree is exported
from the commit object (``git archive``) into a clean directory and the patch is applied there,
and the patched classes are built into the run's private build directory. The same verification then applies against the patched sources: a fresh ``javac`` of pin + patch
must reproduce every loaded class byte for byte. The manifest records base commit, patch SHA-256
and per-class hashes. Never vendors JSettlers source or the patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'runs' / 'external'
BUILD = EXTERNAL / 'jsettlers-build'
CHECKOUT = EXTERNAL / 'jsettlers2'
JAVA_HOME = EXTERNAL / 'jdk' / 'jdk-17.0.21+9' / 'Contents' / 'Home'
SOURCES = ROOT / 'tools' / 'jsettlers' / 'java'
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


def build(out: Path, build_dir: Path = BUILD) -> Path:
    """Compile our bridge classes into ``out``, which must not exist (never reuse a live class dir)."""
    if out.exists() or out.is_symlink():
        raise BuildIdentityError(f'refusing to build into existing {out}; a live JVM may be loading it')
    out.mkdir(parents=True)
    files = sorted(str(p) for p in SOURCES.rglob('*.java'))
    if not files:
        raise SystemExit('no Java sources under ' + str(SOURCES))
    cmd = [java('javac'), '-nowarn', '-encoding', 'UTF-8', '--release', '17',
           '-cp', classpath(False, build_dir), '-d', str(out), *files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    return out


def new_private_dir(parent: Path = EXTERNAL / 'jsettlers-builds') -> Path:
    """A fresh, uniquely named directory for a build outside any run (CLI use)."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%dT%H%M%S-'), dir=parent)) / 'build'


def check_patch(patch: dict | None) -> tuple[Path, str] | None:
    """The declared patch file must exist and hash to the declared SHA-256 (fail closed)."""
    if not patch:
        return None
    path = Path(patch['file'])
    path = path if path.is_absolute() else ROOT / path
    declared = patch.get('sha256')
    if not declared or len(declared) != 64:
        raise BuildIdentityError(f'patch {path} has no declared sha256')
    if not path.is_file():
        raise BuildIdentityError(f'declared JSettlers patch {path} is missing')
    actual = sha256(path)
    if actual != declared:
        raise BuildIdentityError(f'JSettlers patch {path} sha256 {actual} != declared {declared}')
    return path, actual


def patched_tree(dest: Path, patch_path: Path, checkout: Path = CHECKOUT, pin: str = JSETTLERS_COMMIT) -> Path:
    """Export the pinned commit's ``src/`` (not the working tree) into ``dest`` and apply the patch."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(['git', '-C', str(checkout), 'archive', '--format=tar', pin, 'src'],
                             capture_output=True)
    if archive.returncode != 0:
        raise BuildIdentityError('git archive of the pinned JSettlers commit failed: '
                                 + archive.stderr.decode()[-300:])
    subprocess.run(['tar', '-x', '-C', str(dest)], input=archive.stdout, check=True)
    applied = subprocess.run(['git', 'apply', '--whitespace=nowarn', str(patch_path)], cwd=dest,
                             capture_output=True, text=True)
    if applied.returncode != 0:
        raise BuildIdentityError(f'patch {patch_path} does not apply to {pin}: {applied.stderr[-300:]}')
    return dest


def compile_jsettlers(src_root: Path, out: Path, jar: Path, java_home: Path = JAVA_HOME) -> None:
    sources = sorted(str(p) for p in (src_root / 'src' / 'main' / 'java').rglob('*.java'))
    result = subprocess.run([java('javac', java_home), '-nowarn', '-encoding', 'UTF-8', '-cp', str(jar),
                             '-d', str(out), *sources], capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildIdentityError(f'compiling JSettlers failed (rc {result.returncode}): '
                                 + (result.stderr + result.stdout)[-500:])


def stage_jsettlers(build_dir: Path, patch: dict | None = None, checkout: Path = CHECKOUT,
                    pin: str = JSETTLERS_COMMIT, java_home: Path = JAVA_HOME, source: Path = BUILD) -> Path:
    """Stage JSettlers classes + Gson into ``build_dir``, which must not exist yet.

    Unpatched: a read-only copy of the shared release build. Patched: pin + declared patch compiled
    here. Checks (patch hash, pinned clean checkout, Gson hash) run before anything is created; a
    later failure leaves only this new private directory behind, never a changed existing one.
    """
    if build_dir.exists() or build_dir.is_symlink():
        raise BuildIdentityError(f'refusing existing build directory {build_dir}; a live JVM may be loading it')
    declared = check_patch(patch)
    check_checkout(checkout, pin)
    jar = source / GSON_JAR
    if not jar.is_file() or sha256(jar) != GSON_SHA256:
        raise BuildIdentityError(f'{jar} is missing or not the pinned Gson jar')
    build_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(jar, build_dir / GSON_JAR)
    if declared is None:
        shutil.copytree(source / 'classes', build_dir / 'classes')
        return build_dir
    (build_dir / 'classes').mkdir()
    with tempfile.TemporaryDirectory(prefix='jsettlers-patched-') as tmp:
        tree = patched_tree(Path(tmp), declared[0], checkout, pin)
        compile_jsettlers(tree, build_dir / 'classes', jar, java_home)
        shutil.copytree(tree / 'src' / 'main' / 'resources', build_dir / 'classes', dirs_exist_ok=True)
    return build_dir


def build_patched_jsettlers(patch: dict, build_dir: Path, checkout: Path = CHECKOUT, pin: str = JSETTLERS_COMMIT,
                            java_home: Path = JAVA_HOME) -> Path:
    """Build pin + declared patch into the new private ``build_dir`` (classes, resources, Gson)."""
    if not patch:
        raise BuildIdentityError('no patch declared')
    return stage_jsettlers(build_dir, patch, checkout, pin, java_home)


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
                     pin: str = JSETTLERS_COMMIT, gson_sha256: str = GSON_SHA256, patch: dict | None = None) -> dict:
    """Fail closed unless the loaded JSettlers classes and jar are the pinned release (+ declared patch)."""
    declared = check_patch(patch)
    commit = check_checkout(checkout, pin)
    jar = build_dir / GSON_JAR
    if not jar.is_file() or sha256(jar) != gson_sha256:
        raise BuildIdentityError(f'{jar} is missing or not the pinned Gson jar')
    classes = build_dir / 'classes'
    loaded = tree_hashes(classes)
    loaded_classes = {k: v for k, v in loaded.items() if k.endswith('.class')}
    loaded_resources = {k: v for k, v in loaded.items() if not k.endswith('.class')}
    with tempfile.TemporaryDirectory(prefix='jsettlers-rebuild-') as tmp:
        src_root = patched_tree(Path(tmp) / 'src-tree', declared[0], checkout, pin) if declared else checkout
        (Path(tmp) / 'out').mkdir()
        try:
            compile_jsettlers(src_root, Path(tmp) / 'out', jar, java_home)
        except BuildIdentityError as exc:
            raise BuildIdentityError('rebuilding pinned JSettlers failed: ' + str(exc)) from exc
        rebuilt = tree_hashes(Path(tmp) / 'out', '*.class')
        compare_trees(rebuilt, loaded_classes, 'loaded JSettlers classes')
        resources = src_root / 'src' / 'main' / 'resources'
        compare_trees({k: sha256(resources / k) for k in loaded_resources if (resources / k).is_file()}
                      | {k: '' for k in loaded_resources if not (resources / k).is_file()},
                      loaded_resources, 'loaded JSettlers resources')
    identity = {'tag': JSETTLERS_TAG, 'commit': commit, 'classes_digest': digest(loaded),
                'class_files': len(loaded_classes), 'resource_files': len(loaded_resources),
                'gson_jar': GSON_JAR, 'gson_sha256': gson_sha256, 'files': loaded,
                'patch_sha256': declared[1] if declared else None}
    if declared:
        identity['patch'] = {'file': str(declared[0]), 'sha256': declared[1], 'base_commit': commit}
    return identity


def java_version(java_home: Path = JAVA_HOME) -> str:
    out = subprocess.run([java('java', java_home), '-version'], capture_output=True, text=True)
    return (out.stderr or out.stdout).strip()


def build_manifest(build_dir: Path, write: bool = True, patch: dict | None = None) -> dict:
    """Stage a private class path in ``build_dir`` (must not exist), verify it, return its identity.

    JSettlers (pinned release, or pin + declared patch) and our bridge classes are all built or
    copied into ``build_dir``; the manifest's ``build_dir`` and ``classpath`` name it.
    """
    version = java_version()
    if JDK_VERSION not in version:
        raise BuildIdentityError(f'JDK is not {JDK_VERSION}: {version.splitlines()[0]}')
    build_dir = Path(os.path.abspath(build_dir))
    stage_jsettlers(build_dir, patch)
    jsettlers = verify_jsettlers(build_dir, patch=patch)
    out = build(build_dir / 'catanai-classes', build_dir)
    ours_src = tree_hashes(SOURCES, '*.java')
    ours_cls = tree_hashes(out)
    manifest = {
        'schema': 1,
        'jdk': {'home': str(JAVA_HOME), 'version': version},
        'jsettlers': jsettlers,
        'bridge_java': {'sources_digest': digest(ours_src), 'sources': ours_src,
                        'classes_digest': digest(ours_cls), 'classes': ours_cls},
        'build_dir': str(build_dir),
        'classpath': classpath(build_dir=build_dir),
    }
    if write:
        with open(build_dir / MANIFEST.name, 'x') as f:
            f.write(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def verify_loaded(manifest: dict, build_dir: Path | None = None) -> None:
    """Re-hash the class path right before launch; any change since the manifest fails closed."""
    build_dir = build_dir or Path(manifest.get('build_dir', BUILD))
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
    parser.add_argument('--config', help="experiment config; its jsettlers.patch (if any) is built and verified")
    args = parser.parse_args()
    patch = json.loads(Path(args.config).read_text()).get('jsettlers', {}).get('patch') if args.config else None
    build_dir = new_private_dir()  # never an existing class path some JVM may be reading
    if args.verify or patch:
        manifest = build_manifest(build_dir, patch=patch)
        print(f'verified JSettlers {manifest["jsettlers"]["commit"]} patch {manifest["jsettlers"]["patch_sha256"]} '
              f'({manifest["jsettlers"]["class_files"]} classes); manifest {build_dir / MANIFEST.name}',
              file=sys.stderr)
    else:
        stage_jsettlers(build_dir)
        print(f'built {build(build_dir / "catanai-classes", build_dir)}', file=sys.stderr)
    if args.dump_boards:
        subprocess.run([java(), '-cp', classpath(build_dir=build_dir), 'catanai.jsettlers.BoardDump',
                        str(args.dump_boards)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
