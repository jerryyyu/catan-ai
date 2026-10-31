from collections import Counter

from catan_ai.engine import Game, Resource


def game_with(card):
    game = Game.new(['a', 'b', 'c', 'd'], 43)
    game.phase = 'actions'
    game.turn = 4
    game.players[0].development.append((card, 0))
    return game


def test_deck_purchase_timing_and_private_points():
    game = game_with('victory_point')
    assert Counter(game.deck) == {'knight': 14, 'victory_point': 5, 'monopoly': 2, 'road_building': 2, 'year_of_plenty': 2}
    game.deck = ['knight']
    game.players[0].resources.update({Resource.GRAIN: 1, Resource.ORE: 1, Resource.WOOL: 1})
    game.apply(0, {'type': 'buy_development'})
    assert not any(a['type'] == 'play_development' for a in game.legal_actions(0))
    assert game.public_state(0)['players'][0]['points'] == 1
    other = game.public_state(1)['players'][0]
    assert other['points'] == 0 and other['development'] is None
    assert other['development_count'] == 2
    assert not game.deck


def test_monopoly_and_one_development_per_turn():
    game = game_with('monopoly')
    game.players[0].development.append(('knight', 0))
    game.players[1].resources[Resource.ORE] = 2
    game.players[2].resources[Resource.ORE] = 3
    game.apply(0, {'type': 'play_development', 'card': 'monopoly', 'resource': 'ore'})
    assert game.players[0].resources[Resource.ORE] == 5
    assert game.players[1].resources[Resource.ORE] == game.players[2].resources[Resource.ORE] == 0
    assert not any(a['type'] == 'play_development' for a in game.legal_actions(0))


def test_plenty_respects_bank_and_takes_two_identical_cards():
    game = game_with('year_of_plenty')
    game.players[1].resources[Resource.BRICK] = 18
    illegal = {'type': 'play_development', 'card': 'year_of_plenty', 'resources': ['brick', 'brick']}
    assert illegal not in game.legal_actions(0)
    game.apply(0, {'type': 'play_development', 'card': 'year_of_plenty', 'resources': ['ore', 'ore']})
    assert game.players[0].resources[Resource.ORE] == 2


def test_knight_before_roll_returns_to_roll_and_awards_army():
    game = game_with('knight')
    game.phase = 'roll'
    game.players[0].knights = 2
    game.apply(0, {'type': 'play_development', 'card': 'knight'})
    assert game.phase == 'robber'
    assert game.players[0].largest_army and game.players[0].points == 2
    game.apply(0, game.legal_actions(0)[0])
    assert game.phase == 'roll'
    assert game.legal_actions(0) == [{'type': 'roll'}]


def test_road_building_is_free_and_sequential():
    game = game_with('road_building')
    game.players[0].settlements.add(0)
    before = dict(game.players[0].resources)
    game.apply(0, {'type': 'play_development', 'card': 'road_building'})
    for _ in range(2):
        game.apply(0, game.legal_actions(0)[0])
    assert len(game.players[0].roads) == 2
    assert game.players[0].resources == before
    assert game.phase == 'actions'


def test_purchased_victory_point_wins_immediately():
    game = game_with('victory_point')
    game.players[0].cities = {0, 1, 2, 3}
    game.deck = ['victory_point']
    game.players[0].resources.update({Resource.GRAIN: 1, Resource.ORE: 1, Resource.WOOL: 1})
    game.apply(0, {'type': 'buy_development'})
    assert game.winner == 0 and game.players[0].points == 10
