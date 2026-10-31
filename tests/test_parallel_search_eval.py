from copy import deepcopy
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys

import pytest

from catan_ai.search_eval import summarize
from catan_ai.experiments import source_files


runner = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           'scripts/parallel_search_eval.py'))
shard_config, assemble = runner['shard_config'], runner['assemble']


def fixture():
    config = {'stage': 'dev', 'candidate': 'planner_robber', 'control': 'tactical',
              'requires_search': False, 'seeds_per_panel': 2,
              'panels': [{'name': 'tactical', 'seed': 100, 'opponents': ['tactical'] * 3}],
              'gates': {'maximum_match_truncations': 0, 'minimum_paired_win_lift': .05,
                        'maximum_p95_decision_seconds': 2,
                        'maximum_step_seconds': 2, 'maximum_turn_seconds': 10}}
    identity = {'config': config, 'source': {'test': 'producer'}, 'host': 'test-host',
                'process_cap': 4}
    shards = {}
    for index in range(2):
        shard = shard_config(config, index)
        rows = []
        for seat in range(4):
            game = {'winner': seat, 'truncated': False, 'search': {}, 'external': {},
                    'decision_seconds': [[.1]] * 4, 'step_seconds': [[.1]] * 4,
                    'turn_timings': [{'owner': seat, 'complete': True, 'seconds': .2}]}
            control = deepcopy(game)
            control['winner'] = (seat + 1) % 4 if index == 0 else seat
            rows.append({'panel': 'tactical', 'seed': 100 + index, 'seat': seat,
                         'candidate': game, 'baseline': control})
        shards[index] = {**identity, 'config': shard, 'process_cap': 1, 'games': rows,
                         'summary': summarize(shard, rows)}
    return identity, shards


def test_aggregate_recomputes_board_clusters_from_every_raw_pair():
    identity, shards = fixture()
    before = deepcopy(identity)
    result = assemble(identity, dict(reversed(list(shards.items()))))
    assert identity == before
    assert result['process_cap'] == 4
    assert result['summary']['panels']['tactical']['seed_cluster_differences'] == [1, 0]
    assert result['summary']['panels']['tactical']['paired_win_lift'] == .5
    assert len(result['games']) == 8


@pytest.mark.parametrize('defect', ['missing', 'extra', 'source', 'host', 'checkpoint',
                                    'config', 'summary', 'duplicate_pair', 'cap'])
def test_incomplete_or_mismatched_shards_cannot_be_published(defect):
    identity, shards = fixture()
    if defect == 'missing':
        del shards[1]
    elif defect == 'extra':
        shards[2] = deepcopy(shards[1])
    elif defect in ('source', 'host', 'checkpoint'):
        shards[1][defect] = 'different'
    elif defect == 'config':
        shards[1]['config']['panels'][0]['seed'] += 1
    elif defect == 'summary':
        shards[1]['summary']['pass'] = not shards[1]['summary']['pass']
    elif defect == 'duplicate_pair':
        shards[1]['games'][1] = deepcopy(shards[1]['games'][0])
    elif defect == 'cap':
        shards[1]['process_cap'] = 2
    with pytest.raises(ValueError):
        assemble(identity, shards)


def test_sharding_keeps_heldout_admission_and_all_opponent_panels():
    identity, _ = fixture()
    config = identity['config']
    config['stage'] = 'heldout'
    with pytest.raises(ValueError, match='DEV admission'):
        shard_config(config, 0)
    config['requires_dev_pass'] = True
    config['panels'].append({'name': 'mixed', 'seed': 500,
                             'opponents': ['builder', 'external', 'expander']})
    result = shard_config(config, 1)
    assert result['requires_dev_pass']
    assert result['stage'] == 'heldout'
    assert [p['seed'] for p in result['panels']] == [101, 501]
    assert [p['opponents'] for p in result['panels']] == [p['opponents'] for p in config['panels']]


def test_interrupted_run_preserves_completed_shards_and_resumes_only_missing(tmp_path, monkeypatch):
    identity, shards = fixture()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(identity['config']))
    output = tmp_path / 'result.json'
    main = runner['main']
    namespace = main.__globals__
    monkeypatch.setitem(namespace, 'source_files', lambda: identity['source'])
    monkeypatch.setitem(namespace, 'model_identity', lambda _: None)
    monkeypatch.setattr(namespace['platform'], 'node', lambda: 'test-host')
    monkeypatch.setattr(sys, 'argv', ['parallel_search_eval.py', '--config', str(config),
                                    '--output', str(output), '--host', 'test-host',
                                    '--process-cap', '2'])
    calls = []
    fail = True

    def child(command, **kwargs):
        assert command[1:3] == ['-m', 'catan_ai.search_eval']
        child_config = json.loads(Path(command[command.index('--config') + 1]).read_text())
        index = child_config['panels'][0]['seed'] - 100
        calls.append(index)
        if fail and index == 1:
            raise subprocess.CalledProcessError(1, command)
        destination = Path(command[command.index('--output') + 1])
        destination.write_text(json.dumps(shards[index]))

    monkeypatch.setattr(namespace['subprocess'], 'run', child)
    with pytest.raises(subprocess.CalledProcessError):
        main()
    saved = (output.with_suffix('.shards') / '0.json').read_bytes()
    assert not output.exists()
    fail = False
    main()
    assert calls.count(0) == 1
    assert calls.count(1) == 2
    assert (output.with_suffix('.shards') / '0.json').read_bytes() == saved
    result = json.loads(output.read_text())
    assert result['summary']['panels']['tactical']['paired_win_lift'] == .5
    assert result['process_cap'] == 2


@pytest.mark.parametrize('defect,reason', [
    ('missing', 'held-out evaluation requires a passing DEV result'),
    ('failed', 'DEV must pass on this source'),
    ('source', 'DEV must pass on this source'),
    ('overlap', 'held-out seeds overlap DEV'),
    ('diagnostic', 'DEV must pass on this source'),
])
def test_real_child_admission_rejects_before_game_journals(tmp_path, defect, reason):
    identity, _ = fixture()
    config = identity['config']
    dev_config = deepcopy(config)
    dev = {'config': dev_config, 'source': source_files(), 'summary': {'pass': True}}
    config['stage'] = 'heldout'
    config['requires_dev_pass'] = True
    if defect == 'failed':
        dev['summary']['pass'] = False
    elif defect == 'source':
        dev['source'] = {'different': 'producer'}
    elif defect == 'diagnostic':
        dev['config']['stage'] = 'diagnostic'
    config_path = tmp_path / 'heldout.json'
    config_path.write_text(json.dumps(config))
    output = tmp_path / 'result.json'
    script = Path(__file__).resolve().parents[1] / 'scripts/parallel_search_eval.py'
    command = [sys.executable, str(script), '--config', str(config_path),
               '--output', str(output), '--host', platform.node(), '--process-cap', '2']
    if defect != 'missing':
        dev_path = tmp_path / 'dev.json'
        dev_path.write_text(json.dumps(dev))
        command.extend(('--dev-result', str(dev_path)))
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    directory = output.with_suffix('.shards')
    logs = list(directory.glob('*.log'))
    assert len(logs) == 2
    assert all(reason in log.read_text() for log in logs)
    assert not list(directory.glob('*.jsonl'))
    assert not any((directory / f'{i}.json').exists() for i in range(2))
    assert not output.exists()
