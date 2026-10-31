from catan_ai.sim import play_game


def test_full_steps_include_policy_and_completed_turns_are_accounted():
    result = play_game(['heuristic'] * 4, 1200004)
    assert not result.truncated
    for decisions, steps in zip(result.decision_seconds, result.step_seconds):
        assert len(decisions) == len(steps)
        assert all(step >= decision >= 0 for decision, step in zip(decisions, steps))
    assert [t['turn'] for t in result.turn_timings] == list(range(result.turns + 1))
    assert all(t['complete'] and t['seconds'] > 0 for t in result.turn_timings)
    assert all(t['owner'] == t['turn'] % 4 for t in result.turn_timings)
    assert result.as_dict()['turn_timings'] == result.turn_timings


def test_turn_limit_does_not_record_a_nonexistent_next_turn():
    result = play_game(['heuristic'] * 4, 1200004, max_turns=2)
    assert result.truncated
    assert [t['turn'] for t in result.turn_timings] == [0, 1]
    assert all(t['complete'] for t in result.turn_timings)
