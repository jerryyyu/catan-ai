"""Opt-in JSettlers seeding patch: declared-patch build verification, seed derivation, run settings."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import jsettlers_build as jb  # noqa: E402
import jsettlers_smoke as smoke  # noqa: E402

SMOKE = json.loads((ROOT / 'experiments' / 'jsettlers-smoke.json').read_text())
DETERMINISM = json.loads((ROOT / 'experiments' / 'jsettlers-determinism.json').read_text())
PATCH = DETERMINISM['jsettlers']['patch']
PATCH_FILE = ROOT / PATCH['file']
HAVE_JDK = Path(jb.java('javac')).exists() and (jb.BUILD / 'classes').is_dir() and (jb.CHECKOUT / '.git').exists()
HAVE_PATCH = HAVE_JDK and PATCH_FILE.is_file() and jb.sha256(PATCH_FILE) == PATCH['sha256']
needs_patch = pytest.mark.skipif(not HAVE_PATCH, reason='needs the local JDK, JSettlers checkout and seeding patch')


# ------------------------------------------------------------------ seed derivation

def test_seed_derivation_matches_the_java_patch_and_is_stable():
    # Logged by the patched server (catanai-seed game=~botsOnly~2 game_seed=...) for base 20261108.
    assert smoke.derive_seed(20261108, '~botsOnly~2', 'game') == 1113100175055680043
    assert smoke.game_seeds(20261108, '~botsOnly~5') == smoke.game_seeds(20261108, '~botsOnly~5')
    assert all(-(1 << 63) <= v < (1 << 63) for v in smoke.game_seeds(7, 'g').values())


def test_seed_streams_are_separated_by_role_game_and_base():
    seeds = smoke.game_seeds(20261108, '~botsOnly~16')
    assert len(set(seeds.values())) == len(smoke.SEED_ROLES)
    assert seeds != smoke.game_seeds(20261108, '~botsOnly~15')
    assert seeds != smoke.game_seeds(20261109, '~botsOnly~16')
    assert len({smoke.derive_seed(1, f'~botsOnly~{i}', 'board') for i in range(1, 513)}) == 512


def test_policy_seed_is_the_bridge_per_game_seed(monkeypatch):
    from tools.jsettlers.jsbridge import decide
    seen = {}
    monkeypatch.setattr(decide, 'make_policy', lambda name, seed: seen.setdefault('seed', seed))
    decide.Bridge('planner_offer_accept', 20261108).policy_for('~botsOnly~3')
    assert seen['seed'] == smoke.policy_seed(20261108, '~botsOnly~3')


# ------------------------------------------------------------------ run settings

def test_unseeded_config_runs_the_pure_release():
    jset = smoke.jsettlers_settings(SMOKE)
    assert jset == {'seed': None, 'all_smart': False, 'patch': None}
    assert smoke.server_props(jset) == []


def test_seeded_settings_and_refusals():
    jset = smoke.jsettlers_settings(DETERMINISM)
    assert smoke.server_props(jset) == ['-Djsettlers.catanai.seed=20261108', '-Djsettlers.catanai.allsmart=Y']
    no_patch = json.loads(json.dumps(DETERMINISM))
    del no_patch['jsettlers']['patch']
    with pytest.raises(smoke.RunRefused, match='patch'):
        smoke.jsettlers_settings(no_patch)
    parallel = json.loads(json.dumps(DETERMINISM))
    parallel['server']['botgames_parallel'] = 4
    with pytest.raises(smoke.RunRefused, match='parallel'):
        smoke.jsettlers_settings(parallel)
    bad = json.loads(json.dumps(DETERMINISM))
    bad['jsettlers']['seed'] = '1'
    with pytest.raises(smoke.RunRefused, match='integer'):
        smoke.jsettlers_settings(bad)


def test_server_log_facts_are_parsed(tmp_path):
    log = tmp_path / 'server.log'
    log.write_text('\n'.join([
        'catanai-config seed=5 allsmart=true',
        'catanai-robot name=robot 1 strategy=SMART',
        'catanai-robot name=extrabot 1 strategy=SMART',
        'catanai-seed game=~botsOnly~1 game_seed=-42',
        '*** robot 1 joined the game ~botsOnly~1 at 10:11:39',
        '*** extrabot 1 joined the game ~botsOnly~1 at 10:11:39',
        'catanai-start game=~botsOnly~1 first=1 seats=robot 1|extrabot 1|| hexes=[1, 2] numbers=[3] '
        'ports=null deck=[9, 8]',
        'catanai-dice game=~botsOnly~1 turn=1 pn=1 dice=3,4',
        'catanai-dice game=~botsOnly~1 roll=1 turn=2 pn=2 dice=6,6',
    ]))
    g = smoke.server_games(log)['~botsOnly~1']
    assert g['game_seed_logged'] == -42 and g['first_player'] == 1
    assert g['seats'] == ['robot 1', 'extrabot 1', '', ''] and g['hexes'] == '[1, 2]' and g['ports'] == 'null'
    assert g['dice'] == ['1:1:3,4', '1:2:2:6,6'] and g['joined'] == ['robot 1', 'extrabot 1']
    assert g['robot_strategies'] == {'robot 1': 'SMART', 'extrabot 1': 'SMART'}
    assert smoke.bot_strategy('droid 1', False) == 'FAST' and smoke.bot_strategy('droid 1', True) == 'SMART'


# ------------------------------------------------------------------ declared patch verification

def test_patch_sha_mismatch_fails_closed_before_any_build(tmp_path, monkeypatch):
    patch = tmp_path / 'p.patch'
    patch.write_text('not the declared patch\n')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('git/javac reached'))
    with pytest.raises(jb.BuildIdentityError, match='sha256'):
        jb.check_patch({'file': str(patch), 'sha256': '0' * 64})
    with pytest.raises(jb.BuildIdentityError, match='sha256'):
        jb.verify_jsettlers(tmp_path, patch={'file': str(patch), 'sha256': '0' * 64})
    with pytest.raises(jb.BuildIdentityError, match='sha256'):
        jb.build_patched_jsettlers({'file': str(patch), 'sha256': '0' * 64}, build_dir=tmp_path / 'b')
    with pytest.raises(jb.BuildIdentityError, match='missing'):
        jb.check_patch({'file': str(tmp_path / 'absent.patch'), 'sha256': '0' * 64})
    with pytest.raises(jb.BuildIdentityError, match='no declared'):
        jb.check_patch({'file': str(patch)})
    assert jb.check_patch(None) is None


@pytest.mark.skipif(not HAVE_JDK, reason='needs the JSettlers checkout')
def test_patch_that_does_not_apply_fails_closed(tmp_path):
    bad = tmp_path / 'bad.patch'
    bad.write_text('--- a/src/main/java/soc/game/SOCGame.java\n+++ b/src/main/java/soc/game/SOCGame.java\n'
                   '@@ -1,1 +1,1 @@\n-no such line\n+x\n')
    with pytest.raises(jb.BuildIdentityError, match='does not apply'):
        jb.patched_tree(tmp_path / 'tree', bad)


def test_patch_file_is_never_tracked_in_this_repo():
    tracked = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith('.patch') or p.endswith('CatanAiSeed.java')]


@needs_patch
def test_patched_build_verifies_and_stale_or_unpatched_classes_fail_closed(tmp_path):
    """Build pin + patch, verify it, then an unpatched or stale class under the declared patch is refused."""
    build = jb.build_patched_jsettlers(PATCH, build_dir=tmp_path / 'patched')
    identity = jb.verify_jsettlers(build, patch=PATCH)
    assert identity['patch_sha256'] == PATCH['sha256'] and identity['commit'] == jb.JSETTLERS_COMMIT
    assert identity['patch']['base_commit'] == jb.JSETTLERS_COMMIT
    assert 'soc/game/CatanAiSeed.class' in identity['files']
    victim = build / 'classes' / 'soc' / 'game' / 'SOCGame.class'
    victim.write_bytes((jb.BUILD / 'classes' / 'soc' / 'game' / 'SOCGame.class').read_bytes())  # unpatched class
    with pytest.raises(jb.BuildIdentityError, match='classes'):
        jb.verify_jsettlers(build, patch=PATCH)
    # The whole pure release under a declared patch is refused too (missing CatanAiSeed, changed classes).
    shutil.rmtree(build / 'classes')
    shutil.copytree(jb.BUILD / 'classes', build / 'classes')
    with pytest.raises(jb.BuildIdentityError, match='classes'):
        jb.verify_jsettlers(build, patch=PATCH)
    # ...while the same pure release still verifies with no patch declared.
    assert jb.verify_jsettlers(build)['patch_sha256'] is None


def test_compare_counts_paired_dice_by_roll_index_even_when_play_diverges(tmp_path):
    def receipt(name, dice, decisions, winner):
        d = tmp_path / name
        d.mkdir()
        (d / 'decisions.jsonl').write_text('\n'.join(json.dumps(
            {'type': 'decision', 'game': 'g', 'decision': x, 'turn_count': i}) for i, x in enumerate(decisions)))
        game = {'game': 'g', 'server_board_sha': 'b', 'board_id': 'b', 'server_first_player': 1, 'seat': 0,
                'players': ['x', 'y'], 'dice': dice, 'dice_sha': smoke.seq_sha(dice), 'winner': winner, 'turns': 9,
                'decision_seq_sha': smoke.seq_sha(decisions)}
        (d / 'receipt.json').write_text(json.dumps({'games': [game], 'artifacts': {'decisions': str(d / 'decisions.jsonl')}}))
        return d / 'receipt.json'
    a = receipt('a', ['0:1:0:3,4', '1:2:1:6,6', '2:3:2:1,1'], ['roll', 'actions'], 0)
    b = receipt('b', ['0:1:0:3,4', '1:3:2:6,6'], ['roll', 'trade_response'], 1)  # different turn/roller, same values
    c = smoke.compare_receipts(a, b)
    assert c['setup_identical'] == 1 and c['setup_and_dice_k_identical'] == 1 and c['end_to_end_identical'] == 0
    assert c['rows'][0]['dice_values_common_prefix'] == 2 and c['rows'][0]['decision_common_prefix'] == 1
    d = receipt('d', ['0:1:0:3,4', '1:2:1:5,6'], ['roll', 'actions'], 0)
    assert smoke.compare_receipts(a, d)['setup_and_dice_k_identical'] == 0
