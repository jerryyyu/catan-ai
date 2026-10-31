from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.observation import Observation
from catan_ai.sim import play_game


def test_styles_are_seed_independent_and_observation_only():
    game = Game.new(['a', 'b', 'c', 'd'], 1200010)
    for name in ('builder', 'expander'):
        a, b = make_policy(name, 1), make_policy(name, 999)
        view = Observation.for_policy(game, 0, a)
        assert a.choose(view, 0) == b.choose(view, 0)
        assert a.choose(view, 0) in view.actions


def test_both_styles_complete_real_mixed_games_reproducibly():
    policies = ['builder', 'expander', 'tactical', 'heuristic']
    a = play_game(policies, 1200011)
    b = play_game(policies, 1200011)
    assert a == b
    assert not a.truncated


def test_builder_tables_expand_beyond_exhausted_development_deck():
    result = play_game(['heuristic', 'builder', 'builder', 'builder'], 1210100)
    assert not result.truncated
    assert result.turns < 2000
