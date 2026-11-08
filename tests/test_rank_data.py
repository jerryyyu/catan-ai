import json
import runpy
from pathlib import Path

import pytest

from catan_ai.experiments import source_files
from catan_ai.parameter_planner import DEFAULT

rank = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/rank_data.py'))


@pytest.fixture(autouse=True)
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'schema': 1, 'weights': DEFAULT, 'source': source_files()}))
    monkeypatch.setenv('CATAN_PARAMETER_MODEL', str(path))


def test_manifest_and_records_fail_closed(tmp_path):
    config = {'policy': 'planner_offer_accept', 'max_turns': 1000, 'seeds': [10, 1], 'decision_rate': 0.,
              'candidates': 2, 'playouts': 1}
    root = tmp_path / 'run'
    identity = rank['bind_manifest'](root, rank['run_identity'](config))
    with pytest.raises(ValueError):
        rank['bind_manifest'](root, rank['run_identity'](dict(config, playouts=2)))
    path = root / 'games' / '10.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seed': 11, 'identity': identity['identity_sha256']}))
    with pytest.raises(ValueError):
        rank['run_game']((10, config, str(path), identity['identity_sha256']))


def _buy_state(deck_order):
    from catan_ai.engine import Game, Resource
    from catan_ai.bots import make_policy
    from catan_ai.observation import Observation
    game = Game.new(list('abcd'), 21)
    bots = [make_policy('planner_offer_accept', i) for i in range(4)]
    while game.phase.startswith('setup') or game.turn < 8 or game.current != 0 or game.phase != 'actions':
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    game.players[0].resources.update({Resource.WOOL: 1, Resource.GRAIN: 1, Resource.ORE: 1})
    deck = [c for c in game.deck]
    top = {'knight': 'victory_point', 'victory_point': 'knight'}[deck_order]
    i = next(k for k in range(len(deck) - 1, -1, -1) if deck[k] == top)
    deck[-1], deck[i] = deck[i], deck[-1]  # same multiset, different next card
    game.deck = deck
    return game, bots[0]


def test_buy_candidate_features_ignore_hidden_deck_order():
    a, bot = _buy_state('knight')
    b, _ = _buy_state('victory_point')
    assert a.deck[-1] != b.deck[-1] and sorted(a.deck) == sorted(b.deck)
    buy = {'type': 'buy_development'}
    assert buy in a.legal_actions(0)
    assert rank['post_features'](a, 0, bot, buy) == rank['post_features'](b, 0, bot, buy)


def test_complete_from_skips_prior_seeds_and_rejects_foreign_roots(tmp_path):
    prior = tmp_path / 'first'
    (prior / 'games').mkdir(parents=True)
    (prior / 'manifest.json').write_text(json.dumps({'identity_sha256': 'aaa'}))
    (prior / 'games' / '10.json').write_text(json.dumps({'seed': 10, 'identity': 'aaa'}))
    config = {'complete_from': {'root': 'first', 'identity_sha256': 'aaa'}}
    assert rank['completed_elsewhere'](tmp_path / 'second', config) == {10}
    with pytest.raises(ValueError):
        rank['completed_elsewhere'](tmp_path / 'second', {'complete_from': {'root': 'first', 'identity_sha256': 'bbb'}})
    (prior / 'games' / '11.json').write_text(json.dumps({'seed': 12, 'identity': 'aaa'}))
    with pytest.raises(ValueError):
        rank['completed_elsewhere'](tmp_path / 'second', config)
