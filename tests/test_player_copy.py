from copy import deepcopy

from catan_ai.engine import Game, Resource


def test_player_copy_preserves_values_and_isolates_mutable_fields():
    game = Game.new(['a', 'b', 'c', 'd'], 2500000)
    player = game.players[0]
    player.resources[Resource.ORE] = 3
    player.roads.add((0, 1))
    player.settlements.add(2)
    player.cities.add(3)
    player.development.append(('knight', 0))
    clone = deepcopy(game)
    assert clone.players == game.players
    assert clone.rng.getstate() == game.rng.getstate()
    copied = clone.players[0]
    copied.resources[Resource.ORE] = 9
    copied.roads.clear()
    copied.settlements.clear()
    copied.cities.clear()
    copied.development.clear()
    clone.rng.random()
    assert player.resources[Resource.ORE] == 3
    assert player.roads == {(0, 1)} and player.settlements == {2} and player.cities == {3}
    assert player.development == [('knight', 0)]
    assert clone.rng.getstate() != game.rng.getstate()


def test_player_copy_honors_memo_aliases_and_future_attributes():
    player = Game.new(['a', 'b', 'c', 'd'], 2500001).players[0]
    player.extra = {'nested': [1], 'self': player}
    graph = [player.resources, player, player]
    cloned = deepcopy(graph)
    assert cloned[1] is cloned[2]
    assert cloned[0] is cloned[1].resources and cloned[0] is not player.resources
    assert cloned[1].extra['self'] is cloned[1]
    cloned[1].extra['nested'].append(2)
    assert player.extra['nested'] == [1]
