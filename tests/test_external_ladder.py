"""External ladder roots are bound to one run identity; resume refuses mismatches
(Codex HOLD @31b6376, P1/P2). No game is played: policies, the comparator install check
and the process pool are stand-ins."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import external_ladder  # noqa: E402

BUDGET = {'decision': 25, 'turn': 120}
CFG = {'name': 't', 'candidate': 'planner_pv2_tiles', 'max_turns': 1000,
       'variants': {'ab2': {'seed': 10, 'seeds': 2, 'budget': {'decision': 2, 'turn': 10}},
                    'ab3': {'seed': 20, 'seeds': 1, 'budget': BUDGET}}}


class _Net:
    def __init__(self, sha):
        self.meta = {'npz_sha256': sha, 'arch': 'tiles'}


class _Policy:
    def __init__(self, name, sha='a' * 64):
        self.model_hash = f'synthetic-{name}'
        self.net = _Net(sha) if name == 'planner_pv2_tiles' else None


@pytest.fixture
def stubs(monkeypatch):
    """Stand-in policies and comparator installs (the real ones need gitignored weights and
    the pinned Catanatron environment); ``state['sha']`` is the stand-in candidate weights hash."""
    import catan_ai.bots as bots
    state = {'sha': 'a' * 64}
    monkeypatch.setattr(bots, 'make_policy', lambda name, seed: _Policy(name, state['sha']))
    monkeypatch.setattr(external_ladder, 'external_identity',
                        lambda variant, python=None: {'commit': 'pin', 'variant': variant}, raising=False)
    return state


class _Pool:
    """Records scheduled jobs; ``write`` stores a synthetic finished record per job instead of playing."""
    scheduled = []
    write = True

    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, jobs):
        for job in jobs:
            _Pool.scheduled.append(job)
            if _Pool.write and Path(job[6]).exists():  # a stored game is reused, as the old play() did
                yield json.loads(Path(job[6]).read_text())
            elif _Pool.write:
                record = {'seed': job[0], 'seat': job[1], 'variant': job[2], 'winner': job[1],
                          'run_identity': job[7] if len(job) > 7 else None, 'external_fallbacks': 0}
                with Path(job[6]).open('x') as stream:
                    json.dump(record, stream)
                yield record
            else:
                yield fn(job)


@pytest.fixture
def pool(monkeypatch):
    _Pool.scheduled, _Pool.write = [], True
    monkeypatch.setattr(external_ladder, 'Pool', _Pool)
    return _Pool


def _main(monkeypatch, cfg, tmp_path, root, *extra):
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['x', '--config', str(cfg_path), '--root', str(root), *extra])
    external_ladder.main()


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_codex_reproduction_refused(tmp_path):
    """The HOLD's reproduction: a cached seed-999/seat-3/ab2 record must not count as seed 100, seat 0, ab3."""
    path = tmp_path / 'g.json'
    path.write_text(json.dumps({'seed': 999, 'seat': 3, 'variant': 'ab2', 'winner': 3}))
    with pytest.raises(ValueError):
        external_ladder.play((100, 0, 'ab3', 'planner_pv2_tiles', BUDGET, 1000, str(path)))
    with pytest.raises(ValueError, match='does not match'):
        external_ladder.play((100, 0, 'ab3', 'planner_pv2_tiles', BUDGET, 1000, str(path), 'f' * 64))


@pytest.mark.parametrize('field,value', [('seed', 999), ('seat', 3), ('variant', 'ab2'), ('run_identity', 'e' * 64)])
def test_reused_record_must_match_seed_seat_variant_identity(tmp_path, field, value):
    good = {'seed': 100, 'seat': 0, 'variant': 'ab3', 'run_identity': 'f' * 64, 'winner': 1}
    job = (100, 0, 'ab3', 'planner_pv2_tiles', BUDGET, 1000, str(tmp_path / 'g.json'), 'f' * 64)
    (tmp_path / 'g.json').write_text(json.dumps(dict(good, **{field: value})))
    with pytest.raises(ValueError, match='does not match'):
        external_ladder.play(job)
    (tmp_path / 'g.json').write_text(json.dumps(good))
    assert external_ladder.play(job) == good


@pytest.mark.parametrize('change', ['config', 'candidate_name', 'candidate_weights', 'turn_cap'])
def test_changed_declaration_refused_before_scheduling(tmp_path, monkeypatch, stubs, pool, change):
    root = tmp_path / 'root'
    _main(monkeypatch, CFG, tmp_path, root)
    assert json.loads((root / 'manifest.json').read_text())['config'] == CFG
    assert len(pool.scheduled) == 4 * 3
    before = _snapshot(root)
    pool.scheduled.clear()
    cfg = CFG
    if change == 'config':
        cfg = dict(CFG, variants=dict(CFG['variants'], ab3=dict(CFG['variants']['ab3'], budget={'decision': 1, 'turn': 5})))
    elif change == 'candidate_name':
        cfg = dict(CFG, candidate='planner_offer_accept')
    elif change == 'candidate_weights':
        stubs['sha'] = 'b' * 64  # same declared candidate, different weights file
    else:
        cfg = dict(CFG, max_turns=120)
    with pytest.raises(ValueError, match='different run identity'):
        _main(monkeypatch, cfg, tmp_path, root)
    assert pool.scheduled == []
    assert _snapshot(root) == before


def test_nonempty_unbound_root_refused(tmp_path, monkeypatch, stubs, pool):
    root = tmp_path / 'root'
    (root / 'ab2').mkdir(parents=True)
    (root / 'ab2' / '10-0.json').write_text(json.dumps({'seed': 10, 'seat': 0, 'variant': 'ab2', 'winner': 0}))
    before = _snapshot(root)
    with pytest.raises(ValueError, match='unbound root'):
        _main(monkeypatch, CFG, tmp_path, root)
    assert pool.scheduled == []
    assert _snapshot(root) == before


def test_undeclared_variant_in_bound_root_refused_before_scheduling(tmp_path, monkeypatch, stubs, pool):
    root = tmp_path / 'root'
    _main(monkeypatch, CFG, tmp_path, root, '--variants', 'ab3')
    (root / 'ab4').mkdir()
    (root / 'ab4' / '10-0.json').write_text(json.dumps({'seed': 10, 'seat': 0, 'variant': 'ab4', 'winner': 0}))
    pool.scheduled.clear()
    with pytest.raises(ValueError, match='not part of this run'):
        _main(monkeypatch, CFG, tmp_path, root, '--variants', 'ab3')
    assert pool.scheduled == []


def test_stored_game_of_other_identity_refused_before_scheduling(tmp_path, monkeypatch, stubs, pool):
    root = tmp_path / 'root'
    _main(monkeypatch, CFG, tmp_path, root)
    path = root / 'ab2' / '10-0.json'
    path.write_text(json.dumps(dict(json.loads(path.read_text()), run_identity='e' * 64)))
    (root / 'ab2' / '11-1.json').unlink()
    pool.scheduled.clear()
    with pytest.raises(ValueError, match='does not match'):
        _main(monkeypatch, CFG, tmp_path, root)
    assert pool.scheduled == []
    (root / 'ab2' / '99-0.json').write_text('{}')
    path.write_text(json.dumps(dict(json.loads(path.read_text()), run_identity=json.loads(
        (root / 'manifest.json').read_text())['identity_sha256'])))
    with pytest.raises(ValueError, match='not a declared game'):
        _main(monkeypatch, CFG, tmp_path, root)
    assert pool.scheduled == []


def test_identity_binds_candidate_weights_engine_and_comparators(monkeypatch, stubs):
    identity = external_ladder.ladder_identity(CFG)
    assert identity['candidate']['value_net']['npz_sha256'] == 'a' * 64
    assert len(identity['candidate']['value_net']['meta_sha256']) == 64
    assert identity['candidate']['model_hash'] == 'synthetic-planner_pv2_tiles'
    assert identity['engine_backend'] in ('python', 'cython')
    assert 'engine_backend' in identity['source']
    assert set(identity['external']) == {'ab2', 'ab3'}
    stubs['sha'] = 'b' * 64
    assert external_ladder.ladder_identity(CFG)['identity_sha256'] != identity['identity_sha256']


def test_candidate_without_net_records_reason(stubs):
    out = external_ladder.candidate_identity('planner_offer_accept')
    assert out['value_net'] is None and out['value_net_reason'] == 'policy has no value net'


def test_missing_git_metadata_is_flagged(tmp_path):
    (tmp_path / 'COMMIT').write_text('0123abcd\n')
    rev = external_ladder.repo_revision(tmp_path)
    assert rev == {'git_metadata': 'unavailable', 'revision': None, 'dirty': None, 'commit_file': '0123abcd'}
    (tmp_path / 'COMMIT').unlink()
    assert external_ladder.repo_revision(tmp_path)['commit_file'] is None


def test_git_revision_resolved_from_script_repo_not_cwd(tmp_path, monkeypatch):
    repo = Path(external_ladder.__file__).resolve().parents[1]
    if subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True).returncode:
        pytest.skip('source tree is not a git checkout')
    monkeypatch.chdir(tmp_path)
    rev = external_ladder.repo_revision()
    assert rev['git_metadata'] == 'available' and len(rev['revision']) == 40 and isinstance(rev['dirty'], bool)


def test_real_tiles_candidate_identity_hashes_weights():
    pytest.importorskip('numpy')
    from catan_ai.tiles_value import TILES_V1
    if not Path(f'{TILES_V1}.npz').exists():
        pytest.skip('tiles weights are a local artifact')
    meta = json.loads(Path(f'{TILES_V1}.meta.json').read_text())
    out = external_ladder.candidate_identity('planner_pv2_tiles')
    assert out['value_net']['npz_sha256'] == meta['npz_sha256']


def test_matching_resume_reuses_records_without_playing(tmp_path, monkeypatch, stubs, pool):
    root = tmp_path / 'root'
    _main(monkeypatch, CFG, tmp_path, root)
    before = _snapshot(root)
    import catan_ai.engine as engine

    def no_game(*args, **kwargs):
        raise AssertionError('a stored game was replayed')

    monkeypatch.setattr(engine.Game, 'new', no_game)
    pool.scheduled.clear()
    pool.write = False  # pass jobs to the real play(), which must reuse every record
    _main(monkeypatch, CFG, tmp_path, root)
    assert len(pool.scheduled) == 12
    assert _snapshot(root) == before
