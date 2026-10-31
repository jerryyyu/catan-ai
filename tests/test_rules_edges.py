from catan_ai.engine import Game, Resource


def test_single_claimant_gets_remaining_supply():
    game = Game.new(['a', 'b', 'c', 'd'], 2)
    tile = next(t for t in game.board.tiles if isinstance(t.resource, Resource))
    game.players[0].cities.add(tile.vertices[0])
    game.players[1].resources[tile.resource] = 18
    class Dice:
        def __init__(self):
            self.values = iter([max(1, tile.number - 6), min(6, tile.number - 1)])
        def randint(self, low, high):
            return next(self.values)
    game.rng = Dice()
    game._roll()
    assert game.players[0].resources[tile.resource] == 1
    assert game.bank[tile.resource] == 0


def test_largest_army_tie_keeps_holder_then_transfers():
    game = Game.new(['a', 'b', 'c', 'd'], 2)
    game.turn, game.phase = 10, 'actions'
    game.players[0].knights = 3
    game.players[0].largest_army = True
    game.largest_army_holder = 0
    game.current = 1
    game.players[1].knights = 2
    game.players[1].development = [('knight', 0), ('knight', 0)]
    action = {'type':'play_development', 'card':'knight'}
    game.apply(1, action)
    assert game.largest_army_holder == 0
    game.phase, game.played_development = 'actions', False
    game.turn += 4
    game.apply(1, action)
    assert game.largest_army_holder == 1
    assert not game.players[0].largest_army


def test_discarding_player_cannot_win_out_of_turn():
    game = Game.new(['a', 'b', 'c', 'd'], 2)
    game.phase, game.current = 'discard', 1
    game.robber_actor = 0
    game.discard_remaining = {1: 2}
    game.players[1].cities = {0, 1, 2, 3}
    game.players[1].development = [('victory_point', 0)] * 2
    game.players[1].resources[Resource.ORE] = 8
    game.apply(1, {'type':'discard', 'resource':'ore'})
    assert game.winner is None


def test_old_card_playable_even_when_same_type_bought_today():
    game = Game.new(['a', 'b', 'c', 'd'], 2)
    game.phase, game.turn = 'actions', 5
    game.players[0].development = [('monopoly', 5), ('monopoly', 0)]
    game.apply(0, {'type':'play_development', 'card':'monopoly', 'resource':'ore'})
    assert game.players[0].development == [('monopoly', 5)]
