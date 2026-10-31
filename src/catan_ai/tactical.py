"""Short deterministic turn plans using public geometry and the player's hand."""

from copy import deepcopy
import random

from .bots import HeuristicBot
from .engine import Game, Player


def public_projection(view):
    players = []
    for seat, p in enumerate(view.players):
        player = Player(str(seat))
        player.roads, player.settlements, player.cities = set(p.roads), set(p.settlements), set(p.cities)
        player.knights = p.knights
        player.longest_road_award = seat == view.longest_road_holder
        player.largest_army = seat == view.largest_army_holder
        if seat == view.seat:
            player.resources = dict(p.resources)
            player.development = list(zip(p.development, p.development_turns))
        players.append(player)
    return Game(view.board, players, random.Random(0), current=view.seat, phase=view.phase,
                turn=view.turn, longest_road_holder=view.longest_road_holder,
                largest_army_holder=view.largest_army_holder, played_development=view.played_development,
                free_roads=view.free_roads, return_phase=view.return_phase, record_events=False)


def turn_plan(view, depth=3, node_cap=128):
    """Plan builds after at most one known-legal resource acquisition.

    Opponent hands and future randomness are deliberately absent. Restricting
    acquisitions to the root avoids inventing bank supplies for later trades.
    """
    if view.phase not in {'actions', 'free_roads'}:
        return []
    root = public_projection(view)
    seat = view.seat
    initial_points = root.players[seat].points
    best, best_score = [], (0, 0, 0, 0)
    vertex_scores = {}
    nodes = 0
    def visit(game, actions, path):
        nonlocal best, best_score, nodes
        for action in actions:
            kind = action['type']
            build = kind in {'build_city', 'build_settlement', 'build_road'}
            acquisition = (kind == 'bank_trade' or kind == 'play_development'
                           and action['card'] in {'year_of_plenty', 'road_building'})
            if not build and not (not path and acquisition):
                continue
            if nodes >= node_cap:
                return
            nodes += 1
            child = deepcopy(game, {id(game.board): game.board})
            child.apply(seat, action)
            plan = path + [action]
            gain = child.players[seat].points - initial_points
            production = 0
            if gain > 0:
                player = child.players[seat]
                for v in player.settlements | player.cities:
                    if v not in vertex_scores:
                        vertex_scores[v] = HeuristicBot._vertex_score(view, v)
                    production += vertex_scores[v] * (2 if v in player.cities else 1)
            score = (int(child.winner == seat), gain, -len(plan), production)
            if gain > 0 and score > best_score:
                best, best_score = plan, score
            if len(plan) < depth and child.winner is None:
                visit(child, ordered(child.legal_actions(seat)), plan)
    def ordered(actions):
        priority = {'build_city': 0, 'build_settlement': 1, 'play_development': 2,
                    'bank_trade': 3, 'build_road': 4}
        return sorted(actions, key=lambda a: priority.get(a['type'], 5))
    visit(root, ordered(view.actions), [])
    return deepcopy(best)


class TacticalBot(HeuristicBot):
    def choose(self, game, seat):
        plan = turn_plan(game)
        if plan:
            return plan[0]
        return super().choose(game, seat)
