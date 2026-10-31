import math
import pytest
from catan_ai.experiments import cluster_interval, run


def test_cluster_bound_uses_boards_not_individual_games():
    rows = [{'seed': s, 'candidate_seat': seat, 'won': seat < 2, 'truncated': False}
            for s in range(100) for seat in range(4)]
    low, high = cluster_interval(rows)
    margin = math.sqrt(math.log(40) / 200)
    assert low == pytest.approx(.5 - margin)
    assert high == pytest.approx(.5 + margin)
    rows[0]['truncated'] = True
    assert cluster_interval(rows) is None


def test_duplicate_seats_are_rejected():
    with pytest.raises(ValueError):
        cluster_interval([{'seed': 1, 'candidate_seat': 0, 'won': True, 'truncated': False}] * 4)


def test_incomplete_experiment_has_no_strength_claim():
    result = run({'name':'short', 'candidate':'heuristic', 'control':'random',
                  'games_per_seat':1, 'seed':1, 'max_turns':1,
                  'pass_bar':{'metric':'win_rate', 'minimum':0}})
    assert not result['summary']['pass']
    assert result['summary']['win_rate'] is None
    assert result['summary']['seed_cluster_95'] is None
    assert 'engine.py' in result['source_files_sha256']
