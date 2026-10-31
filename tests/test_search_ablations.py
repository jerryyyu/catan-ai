from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation
from catan_ai.tactical import TacticalBot


def test_ablations_hold_rollout_ceiling_and_root_control_fixed():
    for name in ('search_fast', 'search_sequence', 'search_alloc'):
        bot = make_policy(name, 1)
        c = bot.config
        assert c.candidates * c.selection_samples + 2 * c.confirmation_samples == 22
        assert isinstance(bot.baseline, TacticalBot)
        assert not c.tactical
        assert c.root_tactical


def test_sequence_candidate_reaches_win_with_only_observed_actions():
    game = Game.new(['a', 'b', 'c', 'd'], 1200020)
    game.phase, game.turn = 'actions', 8
    p = game.players[0]
    p.cities, p.settlements = {0, 2, 4}, {6, 8, 10}
    p.resources[Resource.ORE], p.resources[Resource.GRAIN] = 3, 1
    p.development = [('year_of_plenty', 0)]
    view = Observation.from_game(game, 0)
    bot = make_policy('search_sequence', 1)
    baseline = bot.baseline.choose(view, 0)
    candidates = bot._candidates(view, baseline)
    sequences = [a for a in candidates if isinstance(a, list)]
    assert len(sequences) == 1
    assert 1 < len(sequences[0]) <= 3
    assert sequences[0][0] in view.actions
    assert bot._rollout(game, sequences[0], 0, 1) == 1
    assert game.winner is None
