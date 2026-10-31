import copy
import math

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation
from catan_ai.search import MonteCarloBot, SearchConfig, improvement_lcb


def action_position():
    game = Game.new(['a', 'b', 'c', 'd'], 29)
    bots = [make_policy('heuristic', s) for s in range(4)]
    for _ in range(500):
        view = Observation.from_game(game, game.current)
        if game.phase == 'actions' and len(view.actions) > 1:
            return game, view
        game.apply(game.current, bots[game.current].choose(view, game.current))
    raise AssertionError('no branching action position')


def test_real_search_reproducible_and_does_not_mutate_game():
    game, view = action_position()
    before = copy.deepcopy(game.public_state(view.seat))
    config = SearchConfig(candidates=2, selection_samples=1, confirmation_samples=2,
                          rollout_decisions=25, particles=16)
    a, b = MonteCarloBot(4, config), MonteCarloBot(4, config)
    assert a.choose(view, view.seat) == b.choose(view, view.seat)
    assert game.public_state(view.seat) == before
    assert a.diagnostics[-1]['rollouts'] == 2
    assert a.diagnostics[-1]['reason'] == 'selection_truncated'
    assert a.diagnostics[-1]['chosen'] == a.diagnostics[-1]['baseline']


def test_confirmation_is_fresh_and_can_select_a_better_action(monkeypatch):
    _, view = action_position()
    bot = MonteCarloBot(4, SearchConfig(candidates=2, selection_samples=2,
                                       confirmation_samples=32, particles=16))
    baseline = copy.deepcopy(bot.baseline).choose(view, view.seat)
    seen = []
    def rollout(world, action, seat, seed):
        seen.append(seed)
        return float(action != baseline)
    monkeypatch.setattr(bot, '_rollout', rollout)
    chosen = bot.choose(view, view.seat)
    record = bot.diagnostics[-1]
    assert chosen != baseline
    assert record['reason'] == 'confirmed' and record['lcb'] > 0
    assert set(seen[:4]).isdisjoint(seen[4:])
    assert len(set(seen[4:])) == 32
    assert record['confirmation_differences'] == [1.0] * 32


def test_noisy_or_truncated_confirmation_cannot_override(monkeypatch):
    _, view = action_position()
    bot = MonteCarloBot(4, SearchConfig(candidates=2, selection_samples=1,
                                       confirmation_samples=2, particles=16))
    baseline = copy.deepcopy(bot.baseline).choose(view, view.seat)
    calls = []
    def rollout(world, action, seat, seed):
        calls.append(seed)
        return float(action != baseline) if len(calls) <= 2 else None
    monkeypatch.setattr(bot, '_rollout', rollout)
    assert bot.choose(view, view.seat) == baseline
    assert bot.diagnostics[-1]['reason'] == 'confirmation_truncated'
    assert bot.diagnostics[-1]['lcb'] is None
    assert improvement_lcb([0.0] * 32, .05) < 0
    assert math.isclose(improvement_lcb([1.0] * 32, .05), 1 - math.sqrt(2 * math.log(20) / 32))


def test_unsupported_phase_retains_baseline_and_avoids_sampling():
    game = Game.new(['a', 'b', 'c', 'd'], 0)
    bot = MonteCarloBot(0)
    action = bot.choose(Observation.from_game(game, 0), 0)
    assert action in game.legal_actions(0)
    assert bot.diagnostics[-1]['reason'] == 'unsupported_phase'
    assert bot.diagnostics[-1]['rollouts'] == 0


def test_api_bot_loop_executes_real_search():
    from catan_ai.api import Session, advance_bots
    game, view = action_position()
    assert view.seat != 0
    bot = MonteCarloBot(4, SearchConfig(candidates=2, selection_samples=1,
                                       confirmation_samples=2, rollout_decisions=1500, particles=32))
    policies = {s: make_policy('heuristic', s) for s in (1, 2, 3)}
    policies[view.seat] = bot
    advance_bots(Session(game, 0, policies))
    assert game.current == 0 or game.winner is not None
    assert sum(d['rollouts'] for d in bot.diagnostics) > 0


def test_bank_trade_shortlist_prefers_build_deficit_to_resource_enum_order():
    game = Game.new(['a', 'b', 'c', 'd'], 29)
    game.phase, game.turn = 'actions', 10
    game.players[0].settlements.add(0)
    game.players[0].resources.update({Resource.BRICK: 1, Resource.LUMBER: 1,
                                      Resource.GRAIN: 1, Resource.ORE: 4})
    view = Observation.from_game(game, 0)
    baseline = next(a for a in view.actions if a['type'] == 'build_road')
    bot = MonteCarloBot(4)
    candidates = bot._candidates(view, baseline)
    assert candidates[0] == baseline
    assert len(candidates) <= bot.config.candidates
    assert {'type': 'bank_trade', 'give': 'ore', 'receive': 'wool'} in candidates


def test_mean_variant_selects_positive_gain_without_claiming_confidence(monkeypatch):
    _, view = action_position()
    bot = MonteCarloBot(4, SearchConfig(candidates=2, selection_samples=1,
                                       confirmation_samples=4, particles=16, selection_rule='mean'))
    baseline = copy.deepcopy(bot.baseline).choose(view, view.seat)
    calls = []
    def rollout(world, action, seat, seed):
        calls.append(seed)
        if len(calls) <= 2:
            return float(action != baseline)
        return float(len(calls) == 3)
    monkeypatch.setattr(bot, '_rollout', rollout)
    assert bot.choose(view, view.seat) != baseline
    assert bot.diagnostics[-1]['reason'] == 'mean_selected'
    assert bot.diagnostics[-1]['lcb'] < 0
