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


def test_incremental_card_counting_equals_a_fresh_fold():
    from copy import deepcopy as dc
    from catan_ai import card_planner
    game = _midgame(11)
    shared = game.observed_view_events(0)
    half = shared[:len(shared) // 2]
    card_planner.resource_estimates(half)          # seeds the resumable fold
    resumed = card_planner.resource_estimates(shared)
    card_planner._FOLDS.clear()
    fresh = card_planner.resource_estimates(dc(list(shared)))  # new objects: no reuse possible
    assert resumed == fresh
    resumed[0][next(iter(resumed[0]))] += 99       # callers get their own dicts
    assert card_planner.resource_estimates(shared) == fresh


def test_income_memo_returns_copies():
    bot = make_policy('planner_offer_accept', 1)
    game = _midgame(11)
    first = bot._income(game, 0)
    first[next(iter(first))] += 50
    assert bot._income(game, 0) != first
    assert '_income_memo' not in deepcopy(bot).__dict__


def test_old_observations_cannot_change_later_ones():
    """Codex HOLD on PR28: shared cached events must be read-only (detached observations)."""
    import json as _json
    from catan_ai.engine import Game as G
    g = G.new(list('abcd'), 8)
    g.apply(g.current, g.legal_actions(g.current)[0])
    view = Observation.from_game(g, 0)
    with pytest.raises(TypeError):
        view.events[0]['counts'][1] = 999
    with pytest.raises(TypeError):
        view.events[0]['action']['type'] = 'x'
    with pytest.raises(TypeError):
        view.events[0]['counts'].append(1)
    with pytest.raises(TypeError):
        view.events[0].update({'counts': None})
    later = Observation.from_game(g, 0)
    assert list(later.events) == g.observed_events(0)
    copied = deepcopy(list(view.events))  # consumers that need to edit get plain copies
    copied[0]['counts'][1] = 999
    assert type(copied[0]) is dict and type(copied[0]['counts']) is list
    assert list(Observation.from_game(g, 0).events) == g.observed_events(0)
    assert _json.loads(_json.dumps(list(view.events))) == g.observed_events(0)


def test_fold_resumes_only_on_an_identical_prefix():
    from catan_ai import card_planner
    game = _midgame(11)
    shared = list(game.observed_view_events(0))
    card_planner._FOLDS.clear()
    card_planner.resource_estimates(shared)
    # same first and last event, replaced middle event: must not reuse the stored fold
    swapped = shared[:]
    k = len(swapped) // 2
    replacement = deepcopy(swapped[k])
    replacement['counts'] = [c + 1 for c in replacement['counts']]
    swapped[k] = replacement
    resumed = card_planner.resource_estimates(swapped)
    card_planner._FOLDS.clear()
    assert resumed == card_planner.resource_estimates(deepcopy(swapped))
