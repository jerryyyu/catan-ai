"""Build the Catan AI StacSettlers bridge and verify every external binary it runs with.

StacSettlers (GPLv3, a JSettlers 1.1.16 fork) is never vendored here. A run's class path is staged
into a **private** build directory that must not exist yet (normally inside the run's own output
directory), so no other build or run can change what a live JVM loads:

* ``stac-classes``: the pinned StacSettlers commit's ``src/`` (exported from the commit object with
  ``git archive``, never the working tree) plus the declared Catan AI match patch (file + SHA-256,
  kept in ``runs/external``), compiled with the pinned JDK (``javac --release 8``), plus the
  checkout's resources. ``verify_stac()`` recompiles pin + patch in a temporary directory and
  requires every loaded ``.class`` to match byte for byte (no missing or extra class) and every
  resource to equal the pinned source copy.
* ``lib``: the jars tracked in the pinned commit's ``lib/`` (from the same ``git archive``), each
  checked against its pinned SHA-256. ``deps``: the Maven dependencies of the pinned ``pom.xml``
  (staged once in ``runs/external/stac-deps``), each checked against its pinned SHA-256.
* ``gson``: the pinned Gson jar the JSettlers bridge already uses.
* ``catanai-classes``: our Java (``tools/stac/java``) plus the transport and log classes reused from
  the JSettlers bridge (``PythonBridge.java``, ``DecisionLog.java``), compiled against the staged
  STAC classes.
* ``HumanActionTypePdf.dat``: the human action-type distribution the STAC MCTS agents read from
  their working directory (tracked in the pinned commit; SHA-256 pinned).

``build_manifest()`` returns the identity of all of it (per-file hashes, aggregate digests, JDK,
patch SHA-256, our sources and classes); ``verify_loaded()`` re-hashes everything right before
launch. Nothing shared is deleted or rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsettlers_build as jb  # noqa: E402  (shared JDK pin and hashing helpers; not modified here)

ROOT = jb.ROOT
EXTERNAL = jb.EXTERNAL
CHECKOUT = EXTERNAL / 'stacsettlers'
DEPS = EXTERNAL / 'stac-deps'
JAVA_HOME = jb.JAVA_HOME
JDK_VERSION = jb.JDK_VERSION
SOURCES = ROOT / 'tools' / 'stac' / 'java'
REUSED_SOURCES = [ROOT / 'tools' / 'jsettlers' / 'java' / 'catanai' / 'jsettlers' / name
                  for name in ('PythonBridge.java', 'DecisionLog.java')]
MANIFEST_NAME = 'stac-build-manifest.json'

STAC_REPO = 'https://github.com/ruflab/StacSettlers'
STAC_COMMIT = 'c9e3b66f69e796b1cdd56a508ec32a6ceed1141d'
GSON_JAR = jb.GSON_JAR
GSON_SHA256 = jb.GSON_SHA256
GSON_SOURCE = jb.BUILD / GSON_JAR
PDF_FILE = 'mcts.game.catan.typepdf.HumanActionTypePdf.dat'
PDF_SHA256 = '19b31c173ab7b449b0681576243cdbbede8f46d335307b88d0dc89a32ddfa748'

# lib/ jars of the pinned commit used on the class path (DeepCatan-0.0.1.jar is a git-lfs pointer
# whose object GitHub no longer serves; the patch stubs the three classes that needed it).
LIB_JARS = {
    'JavaBayes.jar': '32c4d97e7a7c012f23d34767b6fb573459d890fdd1f14b40ed9dcefb9b69297a',
    'mdp-library.jar': '1bd57cc46af21b012ace4f0a2f8ca4eb17d22c478413825a8eff056494b72749',
    'weka.jar': '3c2d7bcce61c77943b460c4e3694116ed7646267f9803956ef0cda1c935bb656',
    'MCTS-1.0.jar': '612ac0ac689cae43728cb1e7c62de2160a4b7469c10ad03e0e2792a22f9138d0',
}

# Maven dependencies of the pinned pom.xml (minus DeepCatan), as resolved from Maven Central.
DEP_JARS = {
    'aima-core-0.11.0.jar': '2fe2be394a9497fc809e364e69d515619b88fdba6b7fc7e1b8309805f0e530f1',
    'AppleJavaExtensions-1.4.jar': '5c46876715c80a166849efe7f2de54aaacafc9f33eb33732a0b6dfc6552cb0c2',
    'commons-cli-1.3.jar': 'f8046bdc72b7ff88afb1dff5ff45451df95290c78a639ec7fa40c953ca89cb26',
    'commons-lang3-3.4.jar': '734c8356420cc8e30c795d64fd1fcd5d44ea9d90342a2cc3262c5158fbc6d98b',
    'guava-20.0.jar': '36a666e3b71ae7f0f0dca23654b67e086e6c93d192f60ba5dfd5519db6c288c8',
    'hamcrest-core-1.3.jar': '66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9',
    'jackson-annotations-2.8.0.jar': 'e61b7343aceeb6ecda291d4ef133cd3e765f178c631c357ffd081abab7f15db8',
    'jackson-core-2.8.0.jar': 'a4cc580989a796953dd1f8a8a033343e8f2ace4dd3152d728a0b465e798937c2',
    'jackson-databind-2.8.0.jar': '4ea14d50f8ce4aeb5c4bb5e92d2e86f83d5f2b06b4aabbb716525820bd6cc4c9',
    'javax.servlet-3.0.0.v201112011016.jar': 'a2cc192a076d9effd10becee8aacbe157f0fe2010fd4322e58aaeff198e56dbe',
    'jcommon-1.0.23.jar': '1e670402809484c71ec74d55b40022a4c4939c7911bd39ee5a0cfb3aaf56397c',
    'jeromq-0.3.4.jar': 'fb5f26734f1e00213450d16b94a085261ea3a746c2a50ded3e26329561fc0524',
    'jetty-client-9.0.5.v20130815.jar': '3e822d1978c3cb3e382a35e6c351d9c5d6285347ca76876bf0ca6b10e33377ff',
    'jetty-continuation-9.0.5.v20130815.jar': 'f516f3785483f6b6dfb2e8015a0a05e1a52d5a3ed1c77bcd98ffe8b22c32a095',
    'jetty-http-9.0.5.v20130815.jar': '38c11730e376b95751fe12e8bf0c9a73ed704b9877b5e1e2c37ce5d3d4675d6a',
    'jetty-io-9.0.5.v20130815.jar': '7a0b5ba684ca248451c00670c72d6f57f4f6265f95e59e02867454e953f41a72',
    'jetty-security-9.0.5.v20130815.jar': 'f01b7e851fb247ec25a14bf80ee34d30047d35b8fb0a8a5d97b22fb476d7e07d',
    'jetty-server-9.0.5.v20130815.jar': 'ff713855b0d4371f10c0f462d2a86b70641d01cd0dd8c52a8cf0b7cf427b1c45',
    'jetty-servlet-9.0.5.v20130815.jar': '6688e9180b3cd3195181d494096bee61dae5d1684932541970d5b341431f0f72',
    'jetty-servlets-9.0.5.v20130815.jar': '976625af6c5bfef9d1b21ac8db15c4ca0ccf33ce0fa660c78c414ef4fc057d75',
    'jetty-util-9.0.5.v20130815.jar': '23c4b6f7027fff729e6ee07aa03efa3e49e8ebd9a87986f79d36d1a70335b184',
    'jetty-webapp-9.0.5.v20130815.jar': 'ad69512e8bd7d39ed48e72e4351df13924543d44aeed03bcfc1811159ba86db4',
    'jetty-xml-9.0.5.v20130815.jar': 'df8b4e647c818fe608abb033d350c06f33309658f4fc61b30f61285b0eba491e',
    'jfreechart-1.0.19.jar': '153d077d6399776a45de97c555ad026eb6201d4bd8af86cfce7b8b4ccfa66263',
    'junit-4.11.jar': '90a8e1603eeca48e7e879f3afbc9560715322985f39a274f6f6070b43f9d06fe',
    'log4j-api-2.3.jar': 'd343479d0e3d39733bd0128ad38eaa100f62671fc3dbcfac34e1c812f5d0ef2b',
    'log4j-core-2.3.jar': '6ae3b0cb657e051f97835a6432c2b0f50a651b36b6d4af395bbe9060bb4ef4b2',
    'mybatis-3.3.0.jar': '97f380bfdd92d7355ff93bbfed4275c898c00426e9d2f78030a4806b585e7e82',
    'mysql-connector-java-5.1.34.jar': 'a993466546ad2ca3685349f5bdb1beb82701a6cb68c3a3583fe15af38c483e31',
    'postgresql-9.3-1102-jdbc41.jar': 'c4530047d03bac8295a9c19fbd7b893b5981edbcd8e52e1597fa2385deec272f',
    'servlet-api-3.0.20100224.jar': '426f5472f79131fb73ceb3171770793575c1a7ea8003f098abd9a99440d17019',
    'websocket-api-9.0.5.v20130815.jar': 'daf15ae2d5c4befc7b4ed59b6279e9aa614fb0b2904ca48b1abc8af4030d7b33',
    'websocket-client-9.0.5.v20130815.jar': 'c2d60751ff608acb8ee0af5ec8db429d484e6271ddd8cb03fee0f4b8e413ddf3',
    'websocket-common-9.0.5.v20130815.jar': '6270e49b17782df8e2a2d54e397160973c792bf0df6c30182df3f26d85f5e60c',
    'websocket-server-9.0.5.v20130815.jar': '1cd7be6d89e36e77b798363d8a1829794f6a28673ddc7abf68bdef04c7ae4c18',
    'websocket-servlet-9.0.5.v20130815.jar': '76e32bcecac35b7bd48daebfa253dc432327fac3391d1b90ad510da0b71056ea',
}

BuildIdentityError = jb.BuildIdentityError
sha256, digest, tree_hashes, compare_trees = jb.sha256, jb.digest, jb.tree_hashes, jb.compare_trees


def java(tool: str = 'java') -> str:
    return jb.java(tool, JAVA_HOME)


def check_patch(patch: dict | None) -> tuple[Path, str]:
    """The match patch is required: the file must exist and hash to the declared SHA-256."""
    if not patch:
        raise BuildIdentityError('a STAC run needs the declared Catan AI match patch (stac.patch)')
    declared = jb.check_patch(patch)
    if declared is None:
        raise BuildIdentityError('stac.patch is empty')
    return declared


def check_checkout(checkout: Path = CHECKOUT, pin: str = STAC_COMMIT) -> str:
    """The checkout must be at ``pin`` with no tracked or untracked change in what we export."""
    try:
        head = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildIdentityError(f'cannot read StacSettlers checkout {checkout}: {exc}') from exc
    if head != pin:
        raise BuildIdentityError(f'StacSettlers checkout is at {head}, pinned {pin}')
    status = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=all',
                             '--', 'src', 'lib', 'pom.xml', PDF_FILE], capture_output=True, text=True).stdout.strip()
    if status:
        raise BuildIdentityError('StacSettlers checkout has local changes: ' + status.splitlines()[0])
    return head


def patched_tree(dest: Path, patch_path: Path, checkout: Path = CHECKOUT, pin: str = STAC_COMMIT) -> Path:
    """Export the pinned commit's src/, lib/, pom.xml and PDF file into ``dest`` and apply the patch."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(['git', '-C', str(checkout), 'archive', '--format=tar', pin,
                              'src', 'lib', 'pom.xml', PDF_FILE], capture_output=True)
    if archive.returncode != 0:
        raise BuildIdentityError('git archive of the pinned StacSettlers commit failed: '
                                 + archive.stderr.decode()[-300:])
    subprocess.run(['tar', '-x', '-C', str(dest)], input=archive.stdout, check=True)
    applied = subprocess.run(['git', 'apply', '--whitespace=nowarn', str(patch_path)], cwd=dest,
                             capture_output=True, text=True)
    if applied.returncode != 0:
        raise BuildIdentityError(f'patch {patch_path} does not apply to {pin}: {applied.stderr[-300:]}')
    return dest


def _checked_jars(directory: Path, pins: dict[str, str], what: str) -> list[Path]:
    out = []
    for name, want in sorted(pins.items()):
        path = directory / name
        if not path.is_file():
            raise BuildIdentityError(f'{what} jar {path} is missing')
        got = sha256(path)
        if got != want:
            raise BuildIdentityError(f'{what} jar {name} sha256 {got} != pinned {want}')
        out.append(path)
    return out


def compile_classpath(lib_dir: Path, deps_dir: Path) -> str:
    return ':'.join(str(p) for p in [*(lib_dir / n for n in sorted(LIB_JARS)), *(deps_dir / n for n in sorted(DEP_JARS))])


def compile_stac(tree: Path, out: Path, lib_dir: Path, deps_dir: Path) -> None:
    sources = sorted(str(p) for p in (tree / 'src' / 'main' / 'java').rglob('*.java'))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(sources))
        argfile = f.name
    try:
        result = subprocess.run([java('javac'), '-nowarn', '-encoding', 'UTF-8', '--release', '8',
                                 '-cp', compile_classpath(lib_dir, deps_dir), '-d', str(out), '@' + argfile],
                                capture_output=True, text=True)
    finally:
        os.unlink(argfile)
    if result.returncode != 0:
        raise BuildIdentityError(f'compiling StacSettlers failed (rc {result.returncode}): '
                                 + (result.stderr + result.stdout)[-500:])


def copy_resources(tree: Path, classes: Path) -> None:
    """Resources as the pom packages them: src/main/java/resources -> resources/, src/main/resources/ -> /."""
    java_res = tree / 'src' / 'main' / 'java' / 'resources'
    for p in sorted(java_res.rglob('*')):
        if p.is_file() and p.suffix != '.java':
            dest = classes / 'resources' / p.relative_to(java_res)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.copytree(tree / 'src' / 'main' / 'resources', classes, dirs_exist_ok=True)


def expected_resources(tree: Path) -> dict[str, str]:
    out = {}
    java_res = tree / 'src' / 'main' / 'java' / 'resources'
    for p in sorted(java_res.rglob('*')):
        if p.is_file() and p.suffix != '.java':
            out[str(Path('resources') / p.relative_to(java_res))] = sha256(p)
    res = tree / 'src' / 'main' / 'resources'
    for p in sorted(res.rglob('*')):
        if p.is_file():
            out[str(p.relative_to(res))] = sha256(p)
    return out


def stage_stac(build_dir: Path, patch: dict, checkout: Path = CHECKOUT, pin: str = STAC_COMMIT) -> Path:
    """Stage pin + patch classes, lib jars, deps, Gson and the PDF file into the new ``build_dir``."""
    if build_dir.exists() or build_dir.is_symlink():
        raise BuildIdentityError(f'refusing existing build directory {build_dir}; a live JVM may be loading it')
    patch_path, _ = check_patch(patch)
    check_checkout(checkout, pin)
    if not GSON_SOURCE.is_file() or sha256(GSON_SOURCE) != GSON_SHA256:
        raise BuildIdentityError(f'{GSON_SOURCE} is missing or not the pinned Gson jar')
    _checked_jars(DEPS, DEP_JARS, 'dependency')
    build_dir.mkdir(parents=True, exist_ok=False)
    for sub in ('stac-classes', 'lib', 'deps'):
        (build_dir / sub).mkdir()
    shutil.copy2(GSON_SOURCE, build_dir / GSON_JAR)
    for name in DEP_JARS:
        shutil.copy2(DEPS / name, build_dir / 'deps' / name)
    with tempfile.TemporaryDirectory(prefix='stac-patched-') as tmp:
        tree = patched_tree(Path(tmp), patch_path, checkout, pin)
        for name in LIB_JARS:
            shutil.copy2(tree / 'lib' / name, build_dir / 'lib' / name)
        _checked_jars(build_dir / 'lib', LIB_JARS, 'lib')
        shutil.copy2(tree / PDF_FILE, build_dir / PDF_FILE)
        compile_stac(tree, build_dir / 'stac-classes', build_dir / 'lib', build_dir / 'deps')
        copy_resources(tree, build_dir / 'stac-classes')
    return build_dir


def verify_stac(build_dir: Path, patch: dict, checkout: Path = CHECKOUT, pin: str = STAC_COMMIT) -> dict:
    """Fail closed unless everything staged is pin + declared patch, rebuilt byte for byte."""
    patch_path, patch_sha = check_patch(patch)
    commit = check_checkout(checkout, pin)
    if sha256(build_dir / GSON_JAR) != GSON_SHA256:
        raise BuildIdentityError('staged Gson jar is not the pinned jar')
    _checked_jars(build_dir / 'deps', DEP_JARS, 'dependency')
    _checked_jars(build_dir / 'lib', LIB_JARS, 'lib')
    extra = sorted({p.name for p in (build_dir / 'deps').iterdir()} - set(DEP_JARS)
                   | {p.name for p in (build_dir / 'lib').iterdir()} - set(LIB_JARS))
    if extra:
        raise BuildIdentityError(f'unexpected jars staged: {extra[:3]}')
    if sha256(build_dir / PDF_FILE) != PDF_SHA256:
        raise BuildIdentityError(f'{PDF_FILE} is not the pinned file')
    loaded = tree_hashes(build_dir / 'stac-classes')
    loaded_classes = {k: v for k, v in loaded.items() if k.endswith('.class')}
    loaded_resources = {k: v for k, v in loaded.items() if not k.endswith('.class')}
    with tempfile.TemporaryDirectory(prefix='stac-rebuild-') as tmp:
        tree = patched_tree(Path(tmp) / 'tree', patch_path, checkout, pin)
        (Path(tmp) / 'out').mkdir()
        compile_stac(tree, Path(tmp) / 'out', build_dir / 'lib', build_dir / 'deps')
        compare_trees(tree_hashes(Path(tmp) / 'out', '*.class'), loaded_classes, 'loaded StacSettlers classes')
        compare_trees(expected_resources(tree), loaded_resources, 'loaded StacSettlers resources')
    return {'repo': STAC_REPO, 'commit': commit, 'patch': {'file': str(patch_path), 'sha256': patch_sha,
                                                           'base_commit': commit},
            'patch_sha256': patch_sha, 'classes_digest': digest(loaded), 'class_files': len(loaded_classes),
            'resource_files': len(loaded_resources), 'files': loaded,
            'lib_jars': dict(LIB_JARS), 'dep_jars_digest': digest(DEP_JARS), 'gson_sha256': GSON_SHA256,
            'pdf_sha256': PDF_SHA256}


def classpath(build_dir: Path) -> str:
    """STAC classes first (they override two MCTS-1.0.jar classes), then ours, lib, deps, Gson."""
    parts = [build_dir / 'stac-classes', build_dir / 'catanai-classes',
             *(build_dir / 'lib' / n for n in sorted(LIB_JARS)),
             *(build_dir / 'deps' / n for n in sorted(DEP_JARS)), build_dir / GSON_JAR]
    return ':'.join(str(p) for p in parts)


def bridge_sources() -> list[Path]:
    files = sorted(SOURCES.rglob('*.java')) + REUSED_SOURCES
    if not files or not all(p.is_file() for p in files):
        raise BuildIdentityError('bridge Java sources are missing')
    return files


def build_bridge(build_dir: Path) -> Path:
    out = build_dir / 'catanai-classes'
    if out.exists() or out.is_symlink():
        raise BuildIdentityError(f'refusing to build into existing {out}')
    out.mkdir(parents=True)
    cp = ':'.join([str(build_dir / 'stac-classes'), str(build_dir / GSON_JAR)])
    result = subprocess.run([java('javac'), '-nowarn', '-encoding', 'UTF-8', '--release', '17', '-cp', cp,
                             '-d', str(out), *map(str, bridge_sources())], capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildIdentityError(f'compiling the STAC bridge failed (rc {result.returncode}): '
                                 + (result.stderr + result.stdout)[-800:])
    return out


def build_manifest(build_dir: Path, patch: dict, write: bool = True) -> dict:
    """Stage a private class path in ``build_dir`` (must not exist), verify it, return its identity."""
    version = jb.java_version(JAVA_HOME)
    if JDK_VERSION not in version:
        raise BuildIdentityError(f'JDK is not {JDK_VERSION}: {version.splitlines()[0]}')
    build_dir = Path(os.path.abspath(build_dir))
    stage_stac(build_dir, patch)
    stac = verify_stac(build_dir, patch)
    out = build_bridge(build_dir)
    ours_src = {str(p.relative_to(ROOT)): sha256(p) for p in bridge_sources()}
    ours_cls = tree_hashes(out)
    manifest = {
        'schema': 1,
        'jdk': {'home': str(JAVA_HOME), 'version': version},
        'stac': stac,
        'bridge_java': {'sources_digest': digest(ours_src), 'sources': ours_src,
                        'classes_digest': digest(ours_cls), 'classes': ours_cls},
        'build_dir': str(build_dir),
        'classpath': classpath(build_dir),
        'pdf_file': str(build_dir / PDF_FILE),
    }
    if write:
        with open(build_dir / MANIFEST_NAME, 'x') as f:
            f.write(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def verify_loaded(manifest: dict, build_dir: Path | None = None) -> None:
    """Re-hash the class path right before launch; any change since the manifest fails closed."""
    build_dir = build_dir or Path(manifest['build_dir'])
    compare_trees(manifest['stac']['files'], tree_hashes(build_dir / 'stac-classes'), 'StacSettlers classes')
    compare_trees(manifest['bridge_java']['classes'], tree_hashes(build_dir / 'catanai-classes'), 'bridge classes')
    _checked_jars(build_dir / 'lib', LIB_JARS, 'lib')
    _checked_jars(build_dir / 'deps', DEP_JARS, 'dependency')
    if sha256(build_dir / GSON_JAR) != GSON_SHA256 or sha256(build_dir / PDF_FILE) != PDF_SHA256:
        raise BuildIdentityError('Gson jar or PDF file changed since the build manifest')


def new_private_dir(parent: Path = EXTERNAL / 'stac-builds') -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%dT%H%M%S-'), dir=parent)) / 'build'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='experiment config with stac.patch')
    args = parser.parse_args(argv)
    patch = json.loads(Path(args.config).read_text())['stac']['patch']
    build_dir = new_private_dir()
    manifest = build_manifest(build_dir, patch)
    print(f'verified StacSettlers {manifest["stac"]["commit"]} + patch {manifest["stac"]["patch_sha256"]} '
          f'({manifest["stac"]["class_files"]} classes); manifest {build_dir / MANIFEST_NAME}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
