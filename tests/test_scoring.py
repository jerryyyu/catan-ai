from catan_ai.scoring import award_holder, longest_road
from catan_ai.engine import Game


def test_road_counts_trails_not_branches_and_stops_at_opponents():
    edges = {(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)}
    assert longest_road(edges, set()) == 4
    assert longest_road(edges, {2}) == 2
    assert longest_road(set(), set()) == 0


def test_loop_can_connect_to_a_tail_without_reusing_an_edge():
    assert longest_road({(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)}, set()) == 5


def test_award_ties_preserve_only_an_eligible_incumbent():
    assert award_holder([4, 3, 2, 1], 5, None) is None
    assert award_holder([5, 5, 0, 0], 5, 1) == 1
    assert award_holder([5, 5, 0, 0], 5, None) is None
    assert award_holder([4, 5, 5, 0], 5, 0) is None
    assert award_holder([5, 6, 0, 0], 5, 0) == 1


def test_award_points_and_next_turn_victory():
    game = Game.new(['a', 'b', 'c', 'd'], 42)
    game.players[1].cities = {30, 31, 32, 33}
    game.players[1].roads = {(i, i + 1) for i in range(5)}
    game._update_longest_road()
    assert game.players[1].points == 10
    assert game.public_state(0)['players'][1]['longest_road']
    game.phase = 'actions'
    game.apply(0, {'type': 'end_turn'})
    assert game.winner == 1
    assert game.legal_actions(1) == []
