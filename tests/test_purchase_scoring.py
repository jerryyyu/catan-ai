"""planner_pv2_tiles_buyfix (src/catan_ai/purchase_scoring.py): red-team D1/D3/D4 purchase scoring."""
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip('numpy')

from catan_ai import purchase_scoring as ps  # noqa: E402
from catan_ai import tiles_value as tv  # noqa: E402
from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.experiments import source_files  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from catan_ai.parameter_planner import DEFAULT  # noqa: E402
from catan_ai.pv_search import afterstate_encoding, candidate_sequences  # noqa: E402


class FakeNet:
    def __init__(self, *_):
        self.w = None

    def logits(self, rows):
        x = np.asarray(rows, dtype=float)
        if self.w is None or len(self.w) != x.shape[1]:
            self.w = np.random.default_rng(0).normal(size=x.shape[1]) / np.sqrt(x.shape[1])
        return x @ self.w


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))
    monkeypatch.setattr(tv, 'TilesValueNet', FakeNet)


def _view(own, played):
    players = [SimpleNamespace(development=tuple(own))] + [SimpleNamespace(development=None)] * 3
    events = tuple({'seat': s, 'action': {'type': 'play_development', 'card': c}} for s, c in played)
    return SimpleNamespace(players=players, events=events, turn=7)


def test_deck_prior_removes_own_and_public_cards():
    prior = dict(ps.deck_prior(_view(['knight', 'monopoly'], [(1, 'knight'), (0, 'year_of_plenty'), (2, 'knight')]), 0))
    assert prior == {'knight': 11, 'victory_point': 5, 'road_building': 2, 'year_of_plenty': 1, 'monopoly': 1}
    gone = dict(ps.deck_prior(_view(['monopoly'], [(1, 'monopoly')]), 0))
    assert 'monopoly' not in gone  # both monopolies accounted for


def test_victory_point_purchase_that_reaches_ten_is_a_win():
    game = Game.new(['p'] * 4, 3)
    while game.phase.startswith('setup'):
        seat = game.current
        game.apply(seat, game.legal_actions(seat)[0])
    me = game.players[0]
    me.cities |= set(me.settlements)  # two cities: 4 points
    me.settlements.clear()
    free = [v for v in range(54) if all(v not in p.settlements and v not in p.cities for p in game.players)]
    me.settlements |= set(free[:3])  # three more settlements: 7 points (legality irrelevant for scoring)
    me.development += [('victory_point', 0), ('victory_point', 0)]  # 9 points, three VP cards left in the deck
    assert me.points == 9
    variants = ps.purchase_variants(game, 0, _view([c for c, _ in me.development], []))
    by_card = dict(zip([c for c, _ in ps.deck_prior(_view([c for c, _ in me.development], []), 0)], variants))
    assert by_card['victory_point'][1] is True and by_card['victory_point'][2] is None
    assert all(not won for card, (_, won, _) in by_card.items() if card != 'victory_point')
    assert abs(sum(w for w, _, _ in variants) - 1) < 1e-12


def test_decisions_without_a_purchase_candidate_match_planner_pv2_tiles():
    s = 4300100
    game = Game.new(['planner_pv2_tiles_buyfix'] + ['planner_offer_accept'] * 3, s)
    bots = [make_policy('planner_pv2_tiles_buyfix', s)] + [make_policy('planner_offer_accept', s + i) for i in (1, 2, 3)]
    checked = 0
    while game.winner is None and game.turn < 300:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if seat == 0 and view.phase == 'actions' and any(a['type'] == 'end_turn' for a in view.actions):
            b = bots[0]
            seqs = candidate_sequences(b, view, seat, b.K)
            if not any(seq and seq[-1]['type'] == 'buy_development' for seq in seqs):
                assert b._pick(view, seat) == ps.super_pick(b, view, seat)
                checked += 1
        game.apply(seat, bots[seat].choose(view, seat))
    assert checked > 5
    assert afterstate_encoding  # imported for the comparison path
