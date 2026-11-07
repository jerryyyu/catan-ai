from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.observation import Observation
from catan_ai.placement_search import PlacementSearchPlanner, setup_world


def _setup_views(seed):
    game = Game.new(list('abcd'), seed)
    served = [make_policy('planner_offer_accept', seed * 4 + s) for s in range(4)]
    while game.phase.startswith('setup'):
        s = game.current
        view = Observation.for_policy(game, s, make_policy('planner_placement_search', 2))
        if game.phase == 'setup_settlement':
            yield game, view, s
        game.apply(s, served[s].choose(Observation.for_policy(game, s, served[s]), s))


def test_setup_world_is_the_true_public_setup_state():
    for seed in (3, 8, 21):
        for game, view, s in _setup_views(seed):
            world = setup_world(view)
            assert (world.phase, world.current, world.setup_index, world.robber_tile) == \
                   (game.phase, game.current, game.setup_index, game.robber_tile)
            for mine, true in zip(world.players, game.players):
                assert (mine.settlements, mine.roads, mine.resources) == (true.settlements, true.roads, true.resources)
            assert world.legal_actions(s) == game.legal_actions(s)


def test_served_ranking_leads_with_the_served_choice():
    for seed in (3, 8, 21):
        for game, view, s in _setup_views(seed):
            planner = make_policy('planner_placement_search', 2)
            served = make_policy('planner_offer_accept', 1).choose(view, s)
            assert planner._served_ranking(view, s)[0] == served


def test_scores_do_not_depend_on_workers_and_choice_is_legal():
    game, view, s = next(_setup_views(5))
    one, two = make_policy('planner_placement_search', 2), make_policy('planner_placement_search', 2)
    for p in (one, two):
        p.candidates, p.playouts, p.max_turns = 2, 2, 40
    one.workers, two.workers = 1, 2
    assert one.placement_scores(view, s) == two.placement_scores(view, s)
    assert one.choose(view, s) in view.actions


def test_non_setup_decisions_are_served():
    game = Game.new(list('abcd'), 9)
    served = make_policy('planner_offer_accept', 2)
    while game.phase.startswith('setup'):
        s = game.current
        game.apply(s, served.choose(Observation.for_policy(game, s, served), s))
    s = game.current
    a, b = make_policy('planner_placement_search', 2), make_policy('planner_offer_accept', 2)
    assert a.choose(Observation.for_policy(game, s, a), s) == b.choose(Observation.for_policy(game, s, b), s)


def test_mixed_pool_is_deterministic_and_draws_styles():
    from catan_ai import placement_search as ps
    game, view, s = next(_setup_views(5))
    one, two = make_policy('planner_placement_mix', 2), make_policy('planner_placement_mix', 3)
    for p in (one, two):
        p.candidates, p.playouts, p.max_turns = 2, 3, 40
    one.workers, two.workers = 1, 2
    assert one.placement_scores(view, s) == two.placement_scores(view, s)
    import random
    drawn = {random.Random(k * 7 + 1).choice(ps.PlacementSearchMixPlanner.pool) for k in range(50)}
    assert drawn == set(ps.PlacementSearchMixPlanner.pool)
