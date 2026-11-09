"""Speed-only changes (perf report items 1-4) must not change any value or decision."""
from copy import deepcopy
import json

import pytest

from catan_ai import _engine_py as E
from catan_ai import pv_search as pv
from catan_ai.bots import make_policy
from catan_ai.engine import BACKEND, Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT
from catan_ai.tactical import public_projection


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _views(seed, limit=60):
    game = Game.new(['p'] * 4, seed)
    bots = [make_policy('planner_offer_accept', seed + i) for i in range(4)]
    out = []
    while game.winner is None and game.turn < 400 and len(out) < limit:
        seat = game.current
        view = Observation.for_policy(game, seat, bots[seat])
        if view.phase == 'actions':
            out.append((view, seat))
        game.apply(seat, bots[seat].choose(view, seat))
    return out, bots


def _generic(obj, memo):
    """The pre-optimization deep copy: a recursive copy of every field."""
    result = type(obj).__new__(type(obj))
    memo[id(obj)] = result
    for k, v in obj.__dict__.items():
        if k != '_observed_cache':
            result.__dict__[k] = deepcopy(v, memo)
    return result


@pytest.mark.skipif(BACKEND != 'python', reason='compares Python object internals')
def test_fast_game_copy_equals_a_generic_deep_copy():
    views, _ = _views(4100011, 40)
    for view, _ in views:
        world = public_projection(view)
        fast = deepcopy(world, {id(world.board): world.board})
        slow = _generic(world, {id(world.board): world.board})
        for k in slow.__dict__:
            if k in ('players', 'rng'):
                continue
            assert fast.__dict__[k] == slow.__dict__[k], k
            if isinstance(slow.__dict__[k], (list, dict, set)) and k != 'board':
                assert fast.__dict__[k] is not world.__dict__[k], k  # no aliasing of mutable state
        for a, b in zip(fast.players, world.players):
            assert vars(a) == vars(b) and a is not b and a.resources is not b.resources
        assert fast.rng.getstate() == world.rng.getstate()


@pytest.mark.skipif(BACKEND != 'python', reason='trusted apply lives in the Python engine')
def test_search_copies_skip_revalidation_only_when_not_validating(monkeypatch):
    views, _ = _views(4100012, 10)
    view, seat = views[0]
    world = public_projection(view)
    assert world.record_events is False
    bogus = {'type': 'build_city', 'vertex': 10 ** 6}
    monkeypatch.setattr(E, '_VALIDATE_SEARCH', True)
    with pytest.raises(E.IllegalAction):
        deepcopy(world).apply(seat, bogus)
    game = Game.new(['p'] * 4, 5)
    with pytest.raises(E.IllegalAction):  # real games always validate
        game.apply(game.current, bogus)


def test_served_tree_reuse_equals_a_fresh_search():
    views, bots = _views(4100013, 50)
    checked = 0
    for view, seat in views:
        bot = bots[seat]
        bot.choose(view, seat)
        memo = getattr(bot, '_served_tree', None)
        if memo is None:
            continue
        reused = pv.served_sequences(bot, view, seat)
        assert bot._served_tree is None  # consumed once
        fresh = pv.served_sequences(bot, view, seat)
        assert reused == fresh
        checked += 1
    assert checked > 5
