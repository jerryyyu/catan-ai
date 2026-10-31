from copy import deepcopy
from dataclasses import replace

from catan_ai.bots import make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation, PlayerView


def api_reference(game, seat):
    """The former public-API-backed observation path, retained as a differential oracle."""
    state = game.public_state(seat)
    players = tuple(PlayerView(
        None if p['resources'] is None else {Resource(r): n for r, n in p['resources'].items()},
        p['resource_count'], game.players[p['seat']].public_points,
        frozenset(p['settlements']), frozenset(p['cities']),
        frozenset(tuple(e) for e in p['roads']),
        None if p['development'] is None else tuple(p['development']),
        p['development_count'],
        tuple(turn for _, turn in game.players[p['seat']].development) if p['seat'] == seat else None,
        p['knights'],
    ) for p in state['players'])
    return Observation(seat, game.phase, game.turn, game.board, players,
                       deepcopy(state['trade_offer']), tuple(deepcopy(state['legal_actions'])),
                       {r: game.trade_rate(seat, r) for r in Resource},
                       game.robber_tile, game.rolled, game.setup_index, game.pending_setup_vertex,
                       dict(game.discard_remaining), game.robber_actor, game.longest_road_holder,
                       game.largest_army_holder, game.played_development, game.return_phase,
                       game.free_roads, tuple(state['events']))


def test_direct_observations_match_api_path_during_real_play():
    game = Game.new(['a', 'b', 'c', 'd'], 1200002)
    bots = [make_policy('heuristic', i) for i in range(4)]
    for _ in range(250):
        if game.winner is not None:
            break
        for seat in range(4):
            assert Observation.from_game(game, seat) == api_reference(game, seat)
        seat = game.current
        game.apply(seat, bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat))


def test_history_opt_out_is_explicit_and_changes_no_other_information():
    game = Game.new(['a', 'b', 'c', 'd'], 1200003)
    game.apply(0, game.legal_actions(0)[0])
    full = Observation.from_game(game, 0)
    assert full.events
    assert Observation.for_policy(game, 0, object()) == full
    assert Observation.for_policy(game, 0, make_policy('mc_mean', 4)) == full
    for name in ('heuristic', 'tactical', 'random', 'strategic'):
        assert Observation.for_policy(game, 0, make_policy(name, 4)) == replace(full, events=())
