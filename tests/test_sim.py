from catan_ai.experiments import run, wilson
from catan_ai.sim import play_game


def test_game_is_deterministic():
    policies = ["heuristic", "random", "random", "random"]
    assert play_game(policies, 91, 80) == play_game(policies, 91, 80)


def test_experiment_rotates_candidate_through_every_seat():
    result = run({
        "name": "tiny",
        "candidate": "heuristic",
        "control": "random",
        "games_per_seat": 1,
        "seed": 8,
        "max_turns": 10,
        "pass_bar": {"metric": "win_rate", "minimum": 0.0},
    })
    assert [row["candidate_seat"] for row in result["games"]] == [0, 1, 2, 3]
    assert result["summary"]["scheduled"] == 4


def test_wilson_interval_contains_observed_rate():
    low, high = wilson(60, 100)
    assert low < 0.6 < high
