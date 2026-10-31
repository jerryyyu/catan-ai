from catan_ai.engine import Game, RESOURCES, Resource
import random


def test_nine_ports_have_disjoint_coastal_endpoints():
    for seed in range(20):
        board = Game.new(['a', 'b', 'c', 'd'], seed).board
        assert len(board.ports) == 9
        assert len({v for edge, _ in board.ports for v in edge}) == 18
        assert sum(kind is None for _, kind in board.ports) == 4
        assert {kind for _, kind in board.ports if kind is not None} == set(RESOURCES)
        for edge, _ in board.ports:
            assert sum(set(edge) <= set(t.vertices) for t in board.tiles) == 1


def test_batched_port_rates_match_reference_for_building_subsets():
    rng = random.Random(1476000)
    for seed in range(12):
        game = Game.new(['a', 'b', 'c', 'd'], seed)
        for _ in range(30):
            for seat, player in enumerate(game.players):
                vertices = rng.sample(range(len(game.board.vertices)), 9)
                player.settlements, player.cities = set(vertices[:5]), set(vertices[5:])
                assert game.trade_rates(seat) == {r: game.trade_rate(seat, r) for r in RESOURCES}


def test_port_rates_require_buildings_and_apply_to_payment():
    game = Game.new(['a', 'b', 'c', 'd'], 42)
    generic = next(edge for edge, kind in game.board.ports if kind is None)
    specific, resource = next((edge, kind) for edge, kind in game.board.ports if kind is not None)
    player = game.players[0]
    player.roads.add(generic)
    assert game.trade_rate(0, Resource.BRICK) == 4
    player.settlements.add(generic[0])
    assert all(game.trade_rate(0, r) == 3 for r in RESOURCES)
    player.cities.add(specific[0])
    assert game.trade_rate(0, resource) == 2
    receive = next(r for r in RESOURCES if r != resource)
    player.resources[resource] = 2
    game.phase = 'actions'
    action = {'type': 'bank_trade', 'give': resource.value, 'receive': receive.value}
    assert action in game.legal_actions(0)
    game.apply(0, action)
    assert player.resources[resource] == 0
    assert player.resources[receive] == 1
