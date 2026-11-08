"""Exact score parity across resources, ports, robber state and fitted weights."""
from copy import deepcopy
import json
import math

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT
from catan_ai.planner import WEIGHTS


def reference_value(bot, game, seat):
    """Frozen pre-optimization expression from eb215a9."""
    player = game.players[seat]
    if player.points >= 10:
        return 100000.
    w = bot.weights
    income = bot._income(game, seat)
    production = sum(WEIGHTS[r] * (n + w[7] * math.sqrt(n)) for r, n in income.items())
    hand = sum(WEIGHTS[r] * (1.6 * min(n, 3) + .45 * max(n - 3, 0))
               for r, n in player.resources.items())
    discard = .5 * max(sum(player.resources.values()) - 7, 0)
    ports = sum(max(income[r] - 5, 0) * (4 / game.trade_rate(seat, r) - 1) for r in Resource)
    development = sum(6 if card == 'victory_point' else 3 for card, _ in player.development)
    return (w[0] * player.points + w[1] * production + w[2] * hand - discard
            + w[3] * bot._geometry(game, seat) + w[4] * ports + w[5] * development)


@pytest.fixture
def bot(tmp_path, monkeypatch):
    path = tmp_path / 'parameters.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))
    return make_policy('planner_offer_accept', 12)


def test_scores_match_after_state_and_weight_changes(bot):
    for seed in (11, 31):
        game = Game.new(list('abcd'), seed)
        while game.phase.startswith('setup'):
            seat = game.current
            game.apply(seat, bot.choose(Observation.for_policy(game, seat, bot), seat))
        for seat in range(4):
            # Include all port kinds, cities and high production; legality is
            # irrelevant to this pure score test.
            player = game.players[seat]
            player.settlements = {endpoints[0] for endpoints, _ in game.board.ports[seat:seat + 3]}
            player.cities.add(next(iter(player.settlements)))
            for r in Resource:
                player.resources[r] = seat + 2
            player.development = [('knight', 0), ('victory_point', 0)]
            for scoring in (False, True):
                bot._scoring = scoring
                for tile in range(len(game.board.tiles)):
                    game.robber_tile = tile
                    for coefficient in (0.2, 2., 5.):
                        bot.weights = list(bot.weights)
                        bot.weights[7] = coefficient
                        assert bot._value(game, seat) == reference_value(bot, game, seat)
                        assert bot._value(game, seat) == reference_value(bot, game, seat)


def test_score_cache_is_bounded_and_not_copied(bot):
    game = Game.new(list('abcd'), 1)
    bot._value(game, 0)
    assert bot._production_value_memo
    copied = deepcopy(bot)
    assert '_production_value_memo' not in copied.__dict__
    assert copied._value(game, 0) == bot._value(game, 0)
    for i in range(4100):
        bot.weights[7] = i / 1000
        bot._value(game, 0)
    assert len(bot._production_value_memo) <= 4096
