"""Run hygiene for the JSettlers runner: sealed outputs, verified binaries, bounded bridge, process cleanup."""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import jsettlers_build as jb  # noqa: E402
import jsettlers_smoke as smoke  # noqa: E402

CONFIG = ROOT / 'experiments' / 'jsettlers-smoke.json'
HAVE_JDK = Path(jb.java('javac')).exists() and (jb.BUILD / 'classes').is_dir() and (jb.CHECKOUT / '.git').exists()
needs_jdk = pytest.mark.skipif(not HAVE_JDK, reason='needs the local JDK and JSettlers build under runs/external')


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reaped-but-listed zombie is not running.
    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith('Z')


# ------------------------------------------------------------------ 1. sealed run outputs

def test_existing_output_root_is_refused_before_build_or_launch(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'receipt.json').write_text('{"old": "receipt"}\n')
    (out / 'decisions.jsonl').write_text('{"type": "game_end", "game": "~botsOnly~1"}\n')
    (out / 'server.log').write_text('old server log\n')
    before = _snapshot(out)

    def forbidden(*args, **kwargs):
        raise AssertionError('build or launch reached for an existing output root')
    monkeypatch.setattr(jb, 'build', forbidden)
    monkeypatch.setattr(jb, 'build_manifest', forbidden)
    monkeypatch.setattr(smoke, 'supervise', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    with pytest.raises(smoke.RunRefused, match='refusing existing run output'):
        smoke.main(['--config', str(CONFIG), '--out', str(out)])
    assert _snapshot(out) == before


def test_empty_existing_root_and_symlink_are_refused(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(smoke.RunRefused):
        smoke.allocate_out(empty)
    link = tmp_path / 'link'
    link.symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(smoke.RunRefused):
        smoke.allocate_out(link)
    fresh = smoke.allocate_out(tmp_path / 'fresh')
    assert fresh.is_dir() and not any(fresh.iterdir())


def test_sealed_files_are_never_overwritten(tmp_path):
    path = tmp_path / 'receipt.json'
    smoke.write_exclusive(path, {'a': 1})
    with pytest.raises(FileExistsError):
        smoke.write_exclusive(path, {'a': 2})
    assert json.loads(path.read_text()) == {'a': 1}


def test_config_pin_must_match_the_verified_build(tmp_path, monkeypatch):
    config = json.loads(CONFIG.read_text())
    config['jsettlers']['commit'] = '0' * 40
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(jb, 'build_manifest', lambda *a, **k: pytest.fail('build reached'))
    with pytest.raises(smoke.RunRefused, match='config pins JSettlers'):
        smoke.main(['--config', str(path), '--out', str(tmp_path / 'run')])


def test_summary_never_reports_negative_truncation(tmp_path):
    rows = [{'type': 'game_end', 'game': f'g{i}', 'seat': 0, 'first_player': 0, 'winner': 1} for i in range(3)]
    summary = smoke.summarize(rows, tmp_path / 'missing.jsonl', expected_games=2)
    assert summary['truncated'] == 0 and summary['unexpected_games'] == 1


# ------------------------------------------------------------------ 2. verified external binaries

def _git_repo(path: Path) -> str:
    (path / 'src').mkdir(parents=True)
    (path / 'src' / 'A.java').write_text('class A {}\n')
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
               GIT_COMMITTER_EMAIL='t@t')
    for cmd in (['init', '-q'], ['add', 'src'], ['commit', '-q', '-m', 'x']):
        subprocess.run(['git', '-C', str(path), *cmd], check=True, env=env, capture_output=True)
    return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_checkout_must_be_the_pinned_clean_commit(tmp_path):
    head = _git_repo(tmp_path / 'co')
    assert jb.check_checkout(tmp_path / 'co', head) == head
    with pytest.raises(jb.BuildIdentityError, match='pinned'):
        jb.check_checkout(tmp_path / 'co', jb.JSETTLERS_COMMIT)
    (tmp_path / 'co' / 'src' / 'A.java').write_text('class A { int x; }\n')
    with pytest.raises(jb.BuildIdentityError, match='changes under src'):
        jb.check_checkout(tmp_path / 'co', head)
    subprocess.run(['git', '-C', str(tmp_path / 'co'), 'checkout', '-q', '--', 'src'], check=True)
    (tmp_path / 'co' / 'src' / 'B.java').write_text('class B {}\n')
    with pytest.raises(jb.BuildIdentityError, match='changes under src'):
        jb.check_checkout(tmp_path / 'co', head)


def test_wrong_gson_jar_fails_closed_before_any_build(tmp_path, monkeypatch):
    head = _git_repo(tmp_path / 'co')
    build = tmp_path / 'build'
    (build / 'classes').mkdir(parents=True)
    (build / jb.GSON_JAR).write_bytes(b'not the pinned jar')
    monkeypatch.setattr(subprocess, 'run', _no_javac(subprocess.run))
    with pytest.raises(jb.BuildIdentityError, match='Gson'):
        jb.verify_jsettlers(build, tmp_path / 'co', pin=head)


def _no_javac(real):
    def run(cmd, *args, **kwargs):
        if 'javac' in str(cmd[0]):
            raise AssertionError('javac reached')
        return real(cmd, *args, **kwargs)
    return run


def test_tree_comparison_catches_missing_extra_and_changed_files():
    expected = {'a.class': '1', 'b.class': '2'}
    jb.compare_trees(expected, dict(expected), 'x')
    for loaded in ({'a.class': '1'}, {**expected, 'c.class': '3'}, {'a.class': '1', 'b.class': '9'}):
        with pytest.raises(jb.BuildIdentityError):
            jb.compare_trees(expected, loaded, 'x')


def test_class_path_change_after_manifest_fails_closed(tmp_path):
    build = tmp_path / 'build'
    for sub in ('classes/soc', 'catanai-classes/catanai'):
        (build / sub).mkdir(parents=True)
    (build / 'classes' / 'soc' / 'X.class').write_bytes(b'x')
    (build / 'catanai-classes' / 'catanai' / 'Y.class').write_bytes(b'y')
    (build / jb.GSON_JAR).write_bytes(b'jar')
    manifest = {'jsettlers': {'files': jb.tree_hashes(build / 'classes'), 'gson_sha256': jb.sha256(build / jb.GSON_JAR)},
                'bridge_java': {'classes': jb.tree_hashes(build / 'catanai-classes')}}
    jb.verify_loaded(manifest, build)
    (build / 'classes' / 'soc' / 'X.class').write_bytes(b'stale rebuild')
    with pytest.raises(jb.BuildIdentityError):
        jb.verify_loaded(manifest, build)


@needs_jdk
def test_stale_loaded_jsettlers_class_fails_closed(tmp_path):
    """A loaded class that the pinned source does not reproduce is refused (real rebuild, ~10 s)."""
    import shutil
    build = tmp_path / 'build'
    shutil.copytree(jb.BUILD / 'classes', build / 'classes')
    shutil.copy2(jb.BUILD / jb.GSON_JAR, build / jb.GSON_JAR)
    identity = jb.verify_jsettlers(build)
    assert identity['commit'] == jb.JSETTLERS_COMMIT and identity['class_files'] > 500
    victim = build / 'classes' / 'soc' / 'game' / 'SOCGame.class'
    victim.write_bytes(victim.read_bytes() + b'\0')
    with pytest.raises(jb.BuildIdentityError, match='classes'):
        jb.verify_jsettlers(build)


# ------------------------------------------------------------------ 3. bounded bridge, process cleanup

FAKE_BRIDGE = textwrap.dedent('''
    import subprocess, sys
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    open(sys.argv[1], 'w').write(str(child.pid))
    for line in sys.stdin:  # accepts requests, never replies
        pass
    import time; time.sleep(300)
''')


NEVER_READS = textwrap.dedent('''
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    open(sys.argv[1], 'w').write(str(child.pid))
    time.sleep(300)  # never reads stdin: a large request fills the pipe and blocks the writer
''')


def _run_timeout_check(tmp_path, script_text, timeout_ms, pad_bytes=None):
    classes = jb.build(tmp_path / 'classes')  # never the shared class dir a live run may be using
    script = tmp_path / 'fake_bridge.py'
    script.write_text(script_text)
    pid_file = tmp_path / 'child.pid'
    args = [str(timeout_ms)] + ([str(pad_bytes)] if pad_bytes else [])
    started = time.monotonic()
    result = subprocess.run(
        [jb.java(), '-cp', f'{jb.classpath(False)}:{classes}', f'-Dcatanai.bridge.cmd={sys.executable} {script} {pid_file}',
         f'-Dcatanai.decision.log={tmp_path / "decisions.jsonl"}',
         'catanai.jsettlers.BridgeTimeoutCheck', *args],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 40
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert f'no bridge reply within {timeout_ms} ms' in out['first_error']
    assert out['retired'] and not out['process_alive'] and out['descendants_alive'] == 0
    assert 'bridge retired' in out['second_error'] and out['second_ms'] < 200
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child = int(pid_file.read_text())
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child), 'the stalled bridge left a descendant running'
    log = [json.loads(line) for line in (tmp_path / 'decisions.jsonl').read_text().splitlines()]
    assert [r['type'] for r in log] == ['bridge_retired']
    return out


@needs_jdk
def test_stalled_bridge_request_times_out_and_retires_the_transport(tmp_path):
    out = _run_timeout_check(tmp_path, FAKE_BRIDGE, 1500)
    assert 1400 <= out['first_ms'] < 10000


@needs_jdk
def test_child_that_never_reads_cannot_block_the_request_write(tmp_path):
    """An 8 MiB request to a child that never reads stdin fills the pipe; the write is under the deadline."""
    out = _run_timeout_check(tmp_path, NEVER_READS, 500, pad_bytes=8 * 1024 * 1024)
    assert 450 <= out['first_ms'] < 1000  # within ~2x the deadline


GROUP = textwrap.dedent('''
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    open(sys.argv[1], 'w').write(str(child.pid))
    time.sleep(float(sys.argv[2]))
''')


@pytest.mark.parametrize('parent_sleep,timeout,expect_timeout', [(300, 1.5, True), (0.2, 30, False)])
def test_supervisor_kills_the_whole_process_group(tmp_path, parent_sleep, timeout, expect_timeout):
    script = tmp_path / 'group.py'
    script.write_text(GROUP)
    pid_file = tmp_path / 'child.pid'
    started = time.monotonic()
    rc, timed_out = smoke.supervise([sys.executable, str(script), str(pid_file), str(parent_sleep)],
                                    tmp_path, tmp_path / 'server.log', timeout)
    assert timed_out is expect_timeout
    assert time.monotonic() - started < timeout + 10
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child), 'a descendant outlived the supervised run'
    with pytest.raises(FileExistsError):
        smoke.supervise([sys.executable, '-c', 'pass'], tmp_path, tmp_path / 'server.log', 5)
