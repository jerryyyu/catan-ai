import json

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT
from catan_ai.pv_search import afterstate_encoding, candidate_sequences


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _state(seed=21, deck_top=None):
    game = Game.new(list('abcd'), seed)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 8 or game.current != 0 or game.phase != 'actions':
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    game.players[0].resources.update({Resource.WOOL: 2, Resource.GRAIN: 3, Resource.ORE: 3,
                                      Resource.BRICK: 1, Resource.LUMBER: 1})
    if deck_top:
        deck = list(game.deck)
        i = next(k for k in range(len(deck) - 1, -1, -1) if deck[k] == deck_top)
        deck[-1], deck[i] = deck[i], deck[-1]
        game.deck = deck
    return game, bots[0]


def test_candidates_start_with_end_now_and_include_the_served_first_move():
    game, bot = _state()
    view = Observation.for_policy(game, 0, bot)
    seqs = candidate_sequences(bot, view, 0, 4)
    assert seqs[0] == [] and 2 <= len(seqs) <= 5
    served = make_policy('planner_offer_accept', 0).choose(view, 0)
    assert served['type'] in ('end_turn', 'offer_trade') or any(s and s[0] == served for s in seqs)


def test_afterstate_of_a_purchase_ignores_the_hidden_deck():
    a, bot = _state(deck_top='knight')
    b, _ = _state(deck_top='victory_point')
    assert a.deck[-1] != b.deck[-1]
    seq = [{'type': 'buy_development'}]
    ea = afterstate_encoding(bot, Observation.for_policy(a, 0, bot), 0, seq)
    eb = afterstate_encoding(bot, Observation.for_policy(b, 0, bot), 0, seq)
    assert ea is not None and ea == eb


def test_afterstate_rejects_sequences_illegal_on_the_world():
    game, bot = _state()
    view = Observation.for_policy(game, 0, bot)
    assert afterstate_encoding(bot, view, 0, [{'type': 'build_city', 'vertex': 10 ** 6}]) is None
