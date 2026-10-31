from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import hashlib

import pytest


trainer = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            'scripts/train_paired_value.py'))


def position(delta=1., target=1.):
    return {'board_seed': 1, 'step': 20, 'choices': [
        {'features': [0., 0.], 'mean_win': 0.},
        {'features': [delta, 0.], 'mean_win': target}]}


def test_ridge_matches_closed_form_and_handles_zero_feature():
    model = trainer['fit']([position(2., .5)])
    assert model['scales'] == [2., 1.]
    assert model['weights'] == pytest.approx([.5 / 1.1, 0.])
    assert trainer['score'](model, [2., 0.]) == pytest.approx(.5 / 1.1)


def test_states_keep_equal_weight_with_different_candidate_counts():
    first, second = position(), position(2.)
    second['choices'].extend([deepcopy(second['choices'][1]) for _ in range(2)])
    examples = list(trainer['pairs']([first, second]))
    assert examples[0][2] == .5
    assert sum(row[2] for row in examples[1:]) == pytest.approx(.5)


def test_margin_fallback_and_validation_gates():
    rows = [position(.04, .5), position(.1, .1)] * 24
    report = trainer['evaluate']({'weights': [1., 0.], 'scales': [1., 1.]}, rows)
    assert report['decisions'][0]['choice'] == 0
    assert report['decisions'][1]['choice'] == 1
    assert report['override_fraction'] == .5
    assert report['selected_advantage'] == pytest.approx(.05)
    assert report['pass'] is True
    assert trainer['evaluate']({'weights': [0., 0.], 'scales': [1., 1.]}, rows)['pass'] is False
    assert trainer['evaluate']({'weights': [1., 0.], 'scales': [1., 1.]}, rows[:2])['pass'] is False


def dataset(tmp_path, seed=100, boards=1):
    plan = {'seed': seed, 'boards': boards, 'states_per_board': 4, 'replicates': 4,
            'max_actions': 4, 'process_cap': 1, 'source': {'fixture': 'fixture'},
            'script_sha256': 'fixture'}
    row = position()
    row.update(board_seed=100, seat=0, turn=8, seconds=1.,
               replicate_seeds=list(range(1000000, 1000004)))
    for index, choice in enumerate(row['choices']):
        choice.update(features=[float(index)] * 143,
                      action={'type': 'end_turn' if index == 0 else 'buy_development'},
                      rollouts=[{'winner': 0 if index else 1, 'win': index,
                                 'truncated': False, 'turns': 12} for _ in range(4)])
    board = {'identity': plan, 'board_seed': 100, 'label_truncations': 0, 'rows': [row]}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    return board


def test_loader_accepts_complete_outcomes(tmp_path):
    board = dataset(tmp_path)
    (tmp_path / '100.json').write_text(json.dumps(board))
    rows, identity, hashes = trainer['read_dataset'](tmp_path, 100, 1)
    assert len(rows) == 1 and len(hashes['100']) == 64
    assert identity['replicates'] == 4


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'nan', 'truncated', 'target', 'winner', 'identity', 'seed'])
def test_loader_rejects_invalid_evidence(tmp_path, defect):
    board = dataset(tmp_path)
    row = board['rows'][0]
    choice = row['choices'][1]
    if defect == 'duplicate':
        choice['action'] = row['choices'][0]['action']
    elif defect == 'nan':
        choice['features'][0] = float('nan')
    elif defect == 'truncated':
        choice['rollouts'][0]['truncated'] = True
    elif defect == 'target':
        choice['mean_win'] = .5
    elif defect == 'winner':
        choice['rollouts'][0]['winner'] = 3
    elif defect == 'identity':
        board['identity'] = {}
    elif defect == 'seed':
        row['replicate_seeds'][0] += 1
    if defect != 'missing':
        (tmp_path / '100.json').write_text(json.dumps(board))
    with pytest.raises((ValueError, FileNotFoundError)):
        trainer['read_dataset'](tmp_path, 100, 1)


def test_cli_freezes_before_validation_and_rejects_replacement(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/train_paired_value.py'
    train = tmp_path / 'train'
    train.mkdir()
    template = dataset(train, 1464000, 64)
    for seed in range(1464000, 1464064):
        board = deepcopy(template)
        board['board_seed'] = seed
        board['rows'][0]['board_seed'] = seed
        board['rows'][0]['replicate_seeds'] = list(range(seed * 10000, seed * 10000 + 4))
        (train / f'{seed}.json').write_text(json.dumps(board))
    command = [sys.executable, str(script), '--root', str(tmp_path)]
    result = subprocess.run(command + ['--mode', 'fit'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checkpoint = (tmp_path / 'model.json').read_bytes()
    assert not (tmp_path / 'validation').exists()
    assert json.loads(result.stdout)['checkpoint_sha256'] == hashlib.sha256(checkpoint).hexdigest()
    result = subprocess.run(command + ['--mode', 'fit'], capture_output=True, text=True)
    assert result.returncode != 0 and 'checkpoint already frozen' in result.stderr
    assert (tmp_path / 'model.json').read_bytes() == checkpoint
    result = subprocess.run(command + ['--mode', 'validate', '--checkpoint-sha', 'wrong'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'frozen checkpoint hash' in result.stderr
    assert not (tmp_path / 'validation.json').exists()
    validation = tmp_path / 'validation'
    validation.mkdir()
    template = dataset(validation, 1465000, 16)
    for seed in range(1465000, 1465016):
        board = deepcopy(template)
        board['board_seed'] = seed
        board['rows'] = [deepcopy(template['rows'][0]) for _ in range(4)]
        for index, row in enumerate(board['rows']):
            row.update(board_seed=seed, seat=index, turn=8 + 20 * index, step=20 + 100 * index,
                       replicate_seeds=list(range(seed * 10000 + index * 4,
                                                  seed * 10000 + index * 4 + 4)))
            for choice in row['choices']:
                for outcome in choice['rollouts']:
                    outcome['winner'] = index if outcome['win'] else (index + 1) % 4
        (validation / f'{seed}.json').write_text(json.dumps(board))
    command += ['--mode', 'validate', '--checkpoint-sha', hashlib.sha256(checkpoint).hexdigest()]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'validation.json').read_text())
    assert report['pass'] is True and report['positions'] == 64
    assert report['selected_advantage'] == 1.
    assert (tmp_path / 'model.json').read_bytes() == checkpoint
    readout = (tmp_path / 'validation.json').read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'validation readout already exists' in result.stderr
    assert (tmp_path / 'validation.json').read_bytes() == readout
