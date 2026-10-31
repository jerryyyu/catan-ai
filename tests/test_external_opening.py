from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_setup_delegates_without_consuming_planner_rng(monkeypatch):
    bot = make_policy('planner_external_opening', 31)
    game = Game.new(['a', 'b', 'c', 'd'], 1469901)
    view = Observation.for_policy(game, 0, bot)
    before = bot.rng.getstate()
    called = []
    def choose(observation, seat):
        called.append((observation, seat))
        return observation.actions[-1]
    monkeypatch.setattr(bot.opening, 'choose', choose)
    assert bot.choose(view, 0) == view.actions[-1]
    assert called == [(view, 0)]
    assert bot.rng.getstate() == before
    assert bot.external_diagnostics is bot.opening.external_diagnostics


def test_post_setup_matches_baseline_and_closes_worker(monkeypatch):
    bot = make_policy('planner_external_opening', 31)
    baseline = make_policy('planner_robber', 31)
    game = Game.new(['a', 'b', 'c', 'd'], 1469901)
    game.phase, game.turn = 'actions', 8
    game.players[0].settlements = {0}
    game.players[0].resources.update({Resource.ORE: 3, Resource.GRAIN: 2})
    closed = []
    bot.opening.process = object()
    def close():
        closed.append(True)
        bot.opening.process = None
    monkeypatch.setattr(bot.opening, 'close', close)
    view = Observation.for_policy(game, 0, bot)
    assert bot.choose(view, 0) == baseline.choose(view, 0)
    assert bot.rng.getstate() == baseline.rng.getstate()
    assert closed == [True]
