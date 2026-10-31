from catan_ai.engine import Game, Resource


def test_bank_trade_cannot_take_an_exhausted_resource():
    game = Game.new(['a', 'b', 'c', 'd'], 10)
    game.phase = 'actions'
    game.players[0].resources[Resource.BRICK] = 4
    game.players[1].resources[Resource.ORE] = 19
    trades = [a for a in game.legal_actions(0) if a['type'] == 'bank_trade']
    assert trades
    assert all(a['receive'] != 'ore' for a in trades)
    game.apply(0, trades[0])
    assert game.bank[Resource.BRICK] == 19
    assert sum(game.bank.values()) + sum(sum(p.resources.values()) for p in game.players) == 95


def test_piece_limits_remove_build_choices():
    game = Game.new(['a', 'b', 'c', 'd'], 11)
    player = game.players[0]
    player.roads = set(game.board.edges[:15])
    player.settlements = set(range(5))
    player.cities = set(range(5, 9))
    player.resources = {r: 5 for r in Resource}
    game.phase = 'actions'
    assert not any(a['type'].startswith('build_') for a in game.legal_actions(0))


def test_production_shortage_does_not_favor_first_seat():
    game = Game.new(['a', 'b', 'c', 'd'], 12)
    tile = next(t for t in game.board.tiles if isinstance(t.resource, Resource))
    game.players[0].settlements.add(tile.vertices[0])
    game.players[1].settlements.add(tile.vertices[3])
    game.players[2].resources[tile.resource] = 18
    class Dice:
        def __init__(self):
            self.values = iter((max(1, tile.number - 6), min(6, tile.number - 1)))
        def randint(self, low, high):
            return next(self.values)
    game.rng = Dice()
    game._roll()
    assert game.players[0].resources[tile.resource] == 0
    assert game.players[1].resources[tile.resource] == 0
    assert game.bank[tile.resource] == 1
