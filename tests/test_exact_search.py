"""planner_pv2_exact (src/catan_ai/exact_search.py): turn-exact search over the tiles value.

CI has no tiles weights (a local artifact), so most tests use a deterministic fake value net with the
real feature encoding; one test runs the real weights when present.
"""
import json

import pytest

np = pytest.importorskip('numpy')

from catan_ai import exact_search as es  # noqa: E402
from catan_ai import tiles_value as tv  # noqa: E402
from catan_ai.bots import make_policy  # noqa: E402
from catan_ai.engine import Game  # noqa: E402
from catan_ai.experiments import source_files  # noqa: E402
from catan_ai.observation import Observation  # noqa: E402
from catan_ai.parameter_planner import DEFAULT  # noqa: E402

needs_weights = pytest.mark.skipif(not tv.TILES_V1.with_suffix('.npz').exists(),
                                   reason='tiles weights are a local artifact (not committed)')


class FakeNet:
    """Linear score of the encoder-v2 features with fixed random weights (deterministic)."""

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


@pytest.fixture
def fake_net(monkeypatch):
    monkeypatch.setattr(tv, 'TilesValueNet', FakeNet)


def _decisions(policy, seed, limit, patch=None):
    """(view, seat, bot) at the first ``limit`` PV decisions of a seeded game, seat 0 = ``policy``."""
    game = Game.new([policy, 'planner_offer_accept', 'planner_offer_accept', 'planner_offer_accept'], seed)
    bots = [make_policy(policy, seed)] + [make_policy('planner_offer_accept', seed + i) for i in (1, 2, 3)]
    out = []
    while game.winner is None and len(out) < limit and game.turn < 400:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if seat == 0 and view.phase == 'actions' and any(a['type'] == 'end_turn' for a in view.actions):
            out.append((view, seat))
        game.apply(seat, bots[seat].choose(view, seat))
    return out, bots[0]


def _trace(policy, seed, max_turns=120):
    game = Game.new([policy] + ['planner_offer_accept'] * 3, seed)
    bots = [make_policy(policy, seed)] + [make_policy('planner_offer_accept', seed + i) for i in (1, 2, 3)]
    trace = []
    while game.winner is None and game.turn < max_turns:
        seat = game.current
        action = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
        trace.append((seat, json.dumps(action, sort_keys=True)))
        game.apply(seat, action)
    return trace, game.winner, game.turn


def test_turn_sequences_are_unique_breadth_first_and_capped(fake_net):
    decisions, bot = _decisions('planner_pv2_exact', 11, 30)
    assert decisions
    for view, seat in decisions:
        seqs, nodes, capped = es.turn_sequences(bot, view, seat, 2000, 400)
        lengths = [len(s) for s in seqs]
        assert lengths == sorted(lengths)  # breadth-first: shorter sequences first
        assert len({json.dumps(s, sort_keys=True) for s in seqs}) == len(seqs)
        assert len(seqs) <= 400 and nodes <= 2000
        assert all(a['type'] in es.EXPAND for s in seqs for a in s)
        assert all(a['type'] != 'buy_development' for s in seqs for a in s[:-1])  # a purchase ends a sequence
        small, _, small_capped = es.turn_sequences(bot, view, seat, 3, 400)
        assert len(small) <= 3 and (small_capped or len(small) == len(seqs))


def test_without_extra_sequences_it_plays_exactly_like_planner_pv2_tiles(fake_net, monkeypatch):
    monkeypatch.setattr(es, 'turn_sequences', lambda *a, **k: ([], 0, False))
    assert _trace('planner_pv2_exact', 5) == _trace('planner_pv2_tiles', 5)


def test_pick_is_never_valued_below_the_control_pick(fake_net):
    decisions, bot = _decisions('planner_pv2_exact', 23, 40)
    from catan_ai.pv_search import afterstate_encoding, candidate_sequences

    def value(seq, view, seat):
        x, won = afterstate_encoding(bot, view, seat, seq)
        return float('inf') if won else float(bot._logits([x])[0])

    checked = 0
    for view, seat in decisions:
        pv = [s for s in candidate_sequences(bot, view, seat, bot.K) if afterstate_encoding(bot, view, seat, s)]
        extra, _, _ = es.turn_sequences(bot, view, seat, bot.MAX_NODES, bot.MAX_STATES)
        allseq = pv + extra
        best_pv = max(value(s, view, seat) for s in pv)
        scored = [s for s in allseq if afterstate_encoding(bot, view, seat, s) is not None]
        best_all = max(value(s, view, seat) for s in scored)
        assert best_all >= best_pv
        checked += 1
    assert checked == len(decisions)


def test_full_game_is_legal_and_deterministic(fake_net):
    a = _trace('planner_pv2_exact', 31, max_turns=400)
    b = _trace('planner_pv2_exact', 31, max_turns=400)
    assert a == b  # count budgets, no clocks: identical replay
    assert a[1] is not None or a[2] == 400


@needs_weights
def test_real_weights_smoke():
    decisions, bot = _decisions('planner_pv2_exact', 7, 20)
    for view, seat in decisions:
        action = bot.choose(view, seat)
        assert action['type'] == 'offer_trade' or action in view.legal_actions(seat)
        assert bot.last_search is None or bot.last_search['scored'] >= 1


def test_first_move_must_be_legal_in_the_real_game(fake_net, monkeypatch):
    """exact-screen-v1 aborted: the card-counted world allowed a bank trade the real bank could not fill."""
    decisions, bot = _decisions('planner_pv2_exact', 41, 15)
    illegal = {'type': 'bank_trade', 'give': 'lumber', 'receive': 'grain'}

    def poisoned(planner, view, seat, max_nodes, max_states, worlds=None):
        return [[illegal]], 1, False
    monkeypatch.setattr(es, 'turn_sequences', poisoned)
    monkeypatch.setattr(es, 'afterstate_encoding',
                        lambda planner, view, seat, seq: (None, True) if seq == [illegal] else
                        __import__('catan_ai.pv_search', fromlist=['x']).afterstate_encoding(planner, view, seat, seq))
    for view, seat in decisions:
        action = bot.choose(view, seat)
        assert action['type'] == 'offer_trade' or action in view.legal_actions(seat)
        if illegal not in view.legal_actions(seat):
            assert action != illegal
