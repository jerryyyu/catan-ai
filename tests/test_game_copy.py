from copy import deepcopy
import random

from catan_ai.engine import Game


class CustomRandom(random.Random):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.extra = [1]

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        clone.setstate(self.getstate())
        clone.extra = deepcopy(self.extra, memo)
        clone.used_custom_copy = True
        return clone


def test_game_copy_preserves_rng_gaussian_cache_and_independent_streams():
    game = Game.new(['a', 'b', 'c', 'd'], 2510000)
    game.rng.gauss(0, 1)
    assert game.rng.gauss_next is not None
    clone = deepcopy(game)
    assert clone.rng is not game.rng and clone.rng.getstate() == game.rng.getstate()
    assert clone.rng.gauss(0, 1) == game.rng.gauss(0, 1)
    assert [clone.rng.random() for _ in range(100)] == [game.rng.random() for _ in range(100)]
    before = game.rng.getstate()
    clone.rng.random()
    assert game.rng.getstate() == before and clone.rng.getstate() != before


def test_game_copy_honors_rng_aliases_attributes_cycles_and_existing_memo():
    game = Game.new(['a', 'b', 'c', 'd'], 2510001)
    game.rng.extra = {'owner': game, 'items': [1]}
    clone, rng = deepcopy([game, game.rng])
    assert clone.rng is rng and rng.extra['owner'] is clone
    rng.extra['items'].append(2)
    assert game.rng.extra['items'] == [1]
    rng_first, clone_second = deepcopy([game.rng, game])
    # An RNG copied before its owning game follows Random's standard protocol,
    # which omits arbitrary extra attributes. Honor that pre-existing memo.
    assert clone_second.rng is rng_first
    assert not hasattr(rng_first, 'extra')
    substitute = random.Random(44)
    assert deepcopy(game, {id(game.rng): substitute}).rng is substitute


def test_custom_rng_subclass_keeps_its_type_state_and_attributes():
    game = Game.new(['a', 'b', 'c', 'd'], 2510002)
    game.rng = CustomRandom(77)
    game.rng.random()
    clone = deepcopy(game)
    assert type(clone.rng) is CustomRandom and clone.rng.getstate() == game.rng.getstate()
    assert clone.rng.used_custom_copy
    clone.rng.extra.append(2)
    assert game.rng.extra == [1]
