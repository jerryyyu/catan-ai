from copy import deepcopy
from pathlib import Path
import random
import runpy

from catan_ai.engine import Game, Resource


def test_theft_draws_do_not_shift_dice_and_copies_preserve_streams(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    module = runpy.run_path(str(scripts / 'probe_continuation_streams.py'))
    stream = module['EventStreams'](41)
    expected = random.Random(41)
    for i in range(100):
        for _ in range(i % 7):
            stream.choice(['ore', 'wool', 'wool'])
        assert stream.randint(1, 6) == expected.randint(1, 6)
    cloned = deepcopy(stream)
    assert [cloned.choice(list(range(7))) for _ in range(20)] == [stream.choice(list(range(7))) for _ in range(20)]
    game = Game.new(['a', 'b', 'c', 'd'], 1462901)
    game.phase, game.turn = 'actions', 8
    player = game.players[0]
    player.cities, player.settlements = {0, 2, 4}, {6, 8, 10}
    player.resources.update({Resource.ORE: 3, Resource.GRAIN: 2})
    before = game.public_state(0), game.rng.getstate(), list(game.deck)
    result = module['continuation'](game, {'type': 'build_city', 'vertex': 6}, 0, ['tactical'] * 4, 19)
    assert result['win'] == 1 and result['truncated'] is False
    assert (game.public_state(0), game.rng.getstate(), game.deck) == before
