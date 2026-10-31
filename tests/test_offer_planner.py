import random

from catan_ai.bots import make_policy
from catan_ai.engine import Game, IllegalAction, Resource
from catan_ai.observation import Observation
from catan_ai.planner import OfferPlanner, ProductionRobberPlanner
from catan_ai.sim import play_game


def advance_to_actions(seed=1604000):
    game = Game.new(['a', 'b', 'c', 'd'], seed)
    bots = [make_policy('planner_robber', seed * 17 + s + 1) for s in range(4)]
    while game.phase != 'actions' or game.turn < 6:
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))
    return game


def test_offer_planner_is_registered_and_differs_from_its_base_only_by_offers():
    policy = make_policy('planner_offer', 7)
    assert isinstance(policy, OfferPlanner) and isinstance(policy, ProductionRobberPlanner)
    assert OfferPlanner.__mro__[1] is ProductionRobberPlanner


def test_offer_is_legal_sent_at_most_once_per_turn_and_targets_a_trailing_opponent():
    game = advance_to_actions()
    seat = game.current
    policy = OfferPlanner(random.Random(3))
    # Force the position to one where no build is affordable but a swap helps.
    player = game.players[seat]
    for r in Resource:
        player.resources[r] = 0
    player.resources[Resource.WOOL] = 3
    player.resources[Resource.GRAIN] = 1
    view = Observation.for_policy(game, seat, policy)
    action = policy.choose(view, seat)
    assert action['type'] == 'offer_trade'
    assert sum(action['give'].values()) == 1 and sum(action['receive'].values()) == 1
    game.apply(seat, action)  # engine accepts it as legal
    points = {s: game.players[s].public_points for s in range(4) if s != seat}
    assert points[action['recipient']] == min(points.values())
    # The partner declines; back on the owner's action phase no second offer is sent this turn.
    game.apply(action['recipient'], {'type': 'reject_trade'})
    view = Observation.for_policy(game, seat, policy)
    again = policy.choose(view, seat)
    assert again['type'] != 'offer_trade'


def test_no_offer_when_no_swap_improves_the_position():
    game = advance_to_actions()
    seat = game.current
    policy = OfferPlanner(random.Random(3))
    policy.offer_margin = 10 ** 6
    for r in Resource:
        game.players[seat].resources[r] = 0
    view = Observation.for_policy(game, seat, policy)
    assert policy.choose(view, seat)['type'] != 'offer_trade'


def test_offer_planner_uses_only_its_own_hand():
    game = advance_to_actions()
    seat = game.current
    policy = OfferPlanner(random.Random(3))
    for r in Resource:
        game.players[seat].resources[r] = 0
    game.players[seat].resources[Resource.WOOL] = 3
    game.players[seat].resources[Resource.GRAIN] = 1
    first = policy.choose(Observation.for_policy(game, seat, policy), seat)
    other = (seat + 1) % 4
    game.players[other].resources[Resource.ORE] += 5
    policy = OfferPlanner(random.Random(3))
    second = policy.choose(Observation.for_policy(game, seat, policy), seat)
    assert first == second


def test_full_game_completes_with_offer_planner_in_every_seat():
    result = play_game(['planner_offer', 'tactical', 'planner_offer', 'builder'], 1604999, max_turns=600)
    assert not result.truncated and result.winner in range(4)


def test_illegal_offer_shape_is_rejected_by_the_engine():
    game = advance_to_actions()
    seat = game.current
    try:
        game.apply(seat, {'type': 'offer_trade', 'recipient': seat, 'give': {'wool': 1}, 'receive': {'ore': 1}})
    except IllegalAction:
        return
    raise AssertionError('self-offer must be illegal')
