import json
from types import SimpleNamespace

import pytest

from catan_ai import search_eval


def test_interrupted_screen_preserves_completed_pairs_and_rejects_source_drift(tmp_path, monkeypatch):
    config = {'stage': 'dev', 'candidate': 'heuristic', 'control': 'tactical',
              'requires_search': False, 'seeds_per_panel': 1, 'max_turns': 20,
              'panels': [{'name': 'panel', 'seed': 1, 'opponents': ['random'] * 3}],
              'gates': {'maximum_match_truncations': 0, 'minimum_paired_win_lift': .05,
                        'maximum_p95_decision_seconds': 2}}
    path, output = tmp_path / 'config.json', tmp_path / 'result.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv', ['search_eval', '--config', str(path), '--output', str(output)])
    monkeypatch.setattr(search_eval, 'source_files', lambda: {'engine.py': 'fixed'})
    calls = []

    def fake_game(policies, seed, limit):
        calls.append((policies[:], seed))
        if len(calls) == 3:
            raise RuntimeError('simulated interruption')
        seat = next(s for s, p in enumerate(policies) if p != 'random')
        result = {'winner': seat if policies[seat] == 'heuristic' else (seat + 1) % 4,
                  'truncated': False, 'search': {}, 'decision_seconds': [[.001]] * 4}
        return SimpleNamespace(as_dict=lambda: result)

    monkeypatch.setattr(search_eval, 'play_game', fake_game)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        search_eval.main()
    journal = output.with_suffix('.jsonl')
    completed = journal.read_bytes()
    assert len(completed.splitlines()) == 2
    monkeypatch.setattr(search_eval, 'source_files', lambda: {'engine.py': 'changed'})
    with pytest.raises(ValueError, match='different source/config/host'):
        search_eval.main()
    assert journal.read_bytes() == completed
    monkeypatch.setattr(search_eval, 'source_files', lambda: {'engine.py': 'fixed'})
    search_eval.main()
    assert journal.read_bytes().startswith(completed)
    result = json.loads(output.read_text())
    assert len(result['games']) == 4 and result['summary']['pass']
    assert len(calls) == 9  # Eight completed game calls plus the interrupted attempt.
