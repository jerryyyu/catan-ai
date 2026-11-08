import json
import math

import pytest

from catan_ai import value_leaf
from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catan_ai.parameter_planner import DEFAULT


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    """A test economic checkpoint bound to the active backend (production binding unchanged)."""
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def _parked_or_run(name):
    """The parked learned leaves pin their feature producers' source hashes. On a tree
    where those producers changed (for example a performance-only edit) they must fail
    closed; their behaviour tests then run only from their frozen snapshots."""
    try:
        return make_policy(name, 1)
    except ValueError as exc:
        assert 'feature producers changed' in str(exc)
        pytest.skip(f'{name} fails closed on changed producers (behaviour covered in its frozen snapshot)')


def test_value_leaf_loads_bound_model_and_scores_finitely():
    _parked_or_run('planner_value_leaf')
    bot = make_policy('planner_value_leaf', 1)
    game = Game.new(list('abcd'), 5)
    assert math.isfinite(bot._leaf(bot._world(Observation.from_game(game, 0)), 0))


def test_value_leaf_rejects_a_changed_checkpoint(tmp_path):
    data = json.loads(value_leaf.CHECKPOINT.read_text())
    data['model']['bias'] += 1
    path = tmp_path / 'tampered.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        value_leaf.load_value_leaf(path)


def test_value_leaf_ignores_opponents_hidden_hands():
    _parked_or_run('planner_value_leaf')
    game = Game.new(list('abcd'), 7)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 12:
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    while game.current != 0 or game.phase != 'actions':
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    view = Observation.for_policy(game, 0, bots[0])
    base = make_policy('planner_value_leaf', 1).choose(view, 0)
    for seat in (1, 2, 3):
        hand = game.players[seat].resources
        hand[Resource.WOOL], hand[Resource.ORE] = hand[Resource.ORE], hand[Resource.WOOL]
    assert make_policy('planner_value_leaf', 1).choose(Observation.for_policy(game, 0, bots[0]), 0) == base


def test_scaled_value_leaf_maps_logit_onto_served_units():
    _parked_or_run('planner_value_leaf_scaled')
    bot = make_policy('planner_value_leaf_scaled', 1)
    game = bot._world(Observation.from_game(Game.new(list('abcd'), 5), 0))
    s = bot.scale
    assert s['train_games'] == 600 and s['slope'] > 0
    assert bot._leaf(game, 0) == pytest.approx(s['leaf_mean'] + s['slope'] * (bot._logit(game, 0) - s['logit_mean']))


def test_end_turn_value_leaf_uses_its_own_frozen_model_and_scale():
    _parked_or_run('planner_value_leaf_end')
    end, v1 = make_policy('planner_value_leaf_end', 1), make_policy('planner_value_leaf_scaled', 1)
    assert end.value_model != v1.value_model and end.scale['train_rows'] == 45047
    game = end._world(Observation.from_game(Game.new(list('abcd'), 5), 0))
    s = end.scale
    assert end._leaf(game, 0) == pytest.approx(s['leaf_mean'] + s['slope'] * (end._logit(game, 0) - s['logit_mean']))


def test_parked_leaves_fail_closed_when_producers_change(monkeypatch):
    from catan_ai import value_leaf as vl
    real = vl.source_files
    monkeypatch.setattr(vl, 'source_files', lambda: dict(real(), **{'unified_planner.py': '0' * 64}))
    with pytest.raises(ValueError, match='feature producers changed'):
        vl.load_value_leaf()
