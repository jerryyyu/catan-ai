from pathlib import Path
import runpy


def test_commands_preserve_frozen_admission_and_bounded_concurrency():
    commands = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 'scripts/run_confirmatory_family.py'))['commands']
    plan = {'planning': {'host': 'test-host', 'process_cap': 4}, 'runs': [
        {'config_path': 'tactical-config', 'result': 'tactical-result', 'dev_result': 'tactical-dev'},
        {'config_path': 'external-config', 'result': 'external-result', 'dev_result': 'external-dev'}]}
    result = commands(plan, Path('calibration'), Path('frozen-scripts'))
    assert len(result) == 2
    for command, entry in zip(result, plan['runs'], strict=True):
        flags = dict(zip(command[2::2], command[3::2], strict=True))
        assert flags == {'--config': entry['config_path'], '--output': entry['result'],
                         '--dev-result': entry['dev_result'], '--calibration': 'calibration',
                         '--host': 'test-host', '--process-cap': '4'}
        assert command[1] == 'frozen-scripts/parallel_search_eval.py'


def test_supervisor_resume_skips_completed_run_and_executes_terminal_reader(tmp_path, monkeypatch):
    import json
    import sys
    import pytest
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  'scripts/run_confirmatory_family.py'))
    main = namespace['main']
    env = main.__globals__
    config_path = tmp_path / 'config.json'
    config = {'name': 'fixture'}
    config_path.write_text(json.dumps(config))
    existing = tmp_path / 'existing.json'
    existing.write_text('{}')
    plan = {'source': {'fixture': 'source'}, 'checkpoint': {'fixture': 'model'},
            'planning': {'host': 'host', 'process_cap': 4}, 'runs': [
                {'config': config, 'config_path': str(config_path), 'result': str(existing),
                 'dev_result': 'unused-dev'}]}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('{}')
    output = tmp_path / 'supervisor'
    monkeypatch.setattr(sys, 'argv', ['supervisor', '--plan', str(plan_path),
                                    '--calibration', str(calibration), '--output', str(output)])
    monkeypatch.setitem(env, 'source_files', lambda: plan['source'])
    monkeypatch.setitem(env, 'model_identity', lambda c: plan['checkpoint'])
    calls = []

    def terminal(command, check):
        calls.append(command)
        assert check
        assert command[1].endswith('read_confirmatory_family.py')
        Path(command[-1]).write_text('{"pass": false}')

    monkeypatch.setattr(env['subprocess'], 'run', terminal)
    main()
    assert len(calls) == 1
    assert json.loads((output / 'readout.json').read_text()) == {'pass': False}
    with pytest.raises(FileExistsError, match='already read'):
        main()
    assert len(calls) == 1
