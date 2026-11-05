"""Performance-only changes must not change behavior (shengji rule: differential tests)."""
import json
from copy import deepcopy

import pytest

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _midgame(seed=5):
    game = Game.new(list('abcd'), seed)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 20:
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    return game


def test_geometry_memo_matches_uncached_and_tracks_builds():
    bot = make_policy('planner_offer_accept', 1)
    game = _midgame()
    for seat in range(4):
        assert bot._geometry(game, seat) == bot._geometry_uncached(game, seat)
        assert bot._geometry(game, seat) == bot._geometry_uncached(game, seat)  # memo hit
    seat = 0
    before = bot._geometry(game, seat)
    taken = {v for p in game.players for v in p.settlements | p.cities}
    free = next(v for v in range(len(game.board.neighbors))
                if v not in taken and not any(n in taken for n in game.board.neighbors[v]))
    child = deepcopy(game, {id(game.board): game.board})
    child.players[1].settlements.add(free)  # an opponent build changes occupancy
    assert bot._geometry(child, seat) == bot._geometry_uncached(child, seat)
    assert bot._geometry(game, seat) == before


def test_copies_drop_the_derived_memo():
    bot = make_policy('planner_offer_accept', 1)
    game = _midgame()
    bot._geometry(game, 0)
    assert bot._geometry_memo
    assert '_geometry_memo' not in deepcopy(bot).__dict__


def test_offer_choice_unchanged_by_plan_cache():
    """The cached offer search equals a direct recomputation of every candidate."""
    game = _midgame(7)
    seat = game.current
    bot = make_policy('planner_offer_accept', 1)
    view = Observation.for_policy(game, seat, bot)
    world = bot._world(view)
    cached = bot._best_offer(view, seat)
    # Recompute with the uncached per-recipient hand score.
    base = bot._hand_score(world, seat)
    best, best_gain = None, bot.offer_margin
    from catan_ai.unified_planner import _RESOURCES
    for receive in _RESOURCES:
        for give in _RESOURCES:
            if give == receive:
                continue
            for count in (1, 2):
                if world.players[seat].resources[give] < count:
                    continue
                for recipient in range(4):
                    if recipient == seat or not bot._offer_allowed(world, seat, recipient):
                        continue
                    held = world.players[recipient].resources[receive]
                    chance = min(1., held)
                    if chance <= 0:
                        continue
                    child = deepcopy(world, {id(world.board): world.board})
                    child.players[recipient].resources[receive] = max(held - 1, 0)
                    child.players[recipient].resources[give] += count
                    child.players[seat].resources[receive] += 1
                    child.players[seat].resources[give] -= count
                    gain = chance * (bot._hand_score(child, seat) - base)
                    if gain > best_gain:
                        gain = bot._offer_gain(world, child, seat, recipient, gain, chance)
                    if gain > best_gain:
                        best, best_gain = (recipient, give.value, receive.value, count), gain
    expect = None if best is None else {'type': 'offer_trade', 'recipient': best[0],
                                        'give': {best[1]: best[3]}, 'receive': {best[2]: 1}}
    assert cached == expect


def test_cached_view_events_equal_fresh_events_for_every_viewer():
    game = _midgame(9)
    for viewer in (0, 1, 2, 3, None):
        assert list(game.observed_view_events(viewer)) == game.observed_events(viewer)
    seat = game.current
    action = game.legal_actions(seat)[0]
    game.apply(seat, action)  # cache extends with the new event
    for viewer in (0, 1, 2, 3):
        assert list(game.observed_view_events(viewer)) == game.observed_events(viewer)
    view = Observation.from_game(game, 0)
    assert list(view.events) == game.observed_events(0)


def test_event_cache_rebuilds_on_replaced_history_and_is_not_copied():
    game = _midgame(9)
    game.observed_view_events(0)
    copied = deepcopy(game)
    assert '_observed_cache' not in copied.__dict__
    assert list(copied.observed_view_events(0)) == game.observed_events(0)
    game.events = game.events[:5]  # replaced list: cache must not serve stale rows
    assert list(game.observed_view_events(0)) == game.observed_events(0)
