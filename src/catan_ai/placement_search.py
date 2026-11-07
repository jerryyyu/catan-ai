"""Opening placement search by playouts (Jerry 2026-11-07; follows phase-diagnostic-v1).

At each own ``setup_settlement`` decision the served bot (``planner_offer_accept``)
ranks the legal vertices with its own placement value; the top ``candidates`` are
each played out ``playouts`` times to the end of the game with the served bot in all
four seats, and the vertex with the most wins is placed (ties keep the served order).
Every other decision is the served bot's.

Information: setup has no hidden hands. The world is rebuilt by replaying the public
event history on the observed board; only the development deck order (hidden) and the
dice are sampled, per playout, from the standard deck composition. Playout ``r`` uses
the same random stream for every candidate (common random numbers). The opponents'
identities are not used: every seat in a playout is the served bot.

Playouts run in a process pool (``workers``); results depend only on the streams, not
on the number of workers.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
import random

from .engine import DESERT, Game, Player
from .tactical import public_projection
from .unified_planner import AcceptanceOfferPlanner

DECK = ['knight'] * 14 + ['victory_point'] * 5 + ['monopoly', 'year_of_plenty', 'road_building'] * 2


def setup_world(view):
    """The true setup state, rebuilt from public information only (deck left empty)."""
    board = view.board
    robber = next(i for i, tile in enumerate(board.tiles) if tile.resource == DESERT)
    game = Game(board, [Player(str(s)) for s in range(4)], random.Random(0), robber_tile=robber, deck=[])
    for event in view.events:
        game.apply(event['seat'], dict(event['action']))
    return game


def playout(args):
    """One full game from ``world`` after ``action``; 1 if ``seat`` wins."""
    world, seat, action, stream, max_turns = args
    from .bots import make_policy
    from .observation import Observation
    game = deepcopy(world)
    game.rng = random.Random(stream)
    game.deck = list(DECK)
    game.rng.shuffle(game.deck)
    bots = [make_policy('planner_offer_accept', stream * 4 + s) for s in range(4)]
    game.apply(seat, action)
    while game.winner is None and game.turn < max_turns:
        s = game.current
        game.apply(s, bots[s].choose(Observation.for_policy(game, s, bots[s]), s))
    return int(game.winner == seat)


class PlacementSearchPlanner(AcceptanceOfferPlanner):
    needs_history = True
    candidates = 12
    playouts = 24
    workers = 7
    max_turns = 1000

    def _served_ranking(self, view, seat):
        root = public_projection(view)

        def placement(action):
            child = deepcopy(root, {id(root.board): root.board})
            child.players[seat].settlements.add(action['vertex'])
            return self._value(child, seat)
        legal = view.legal_actions(seat)
        scores = [placement(a) for a in legal]
        # stable: the served bot's max() keeps the first maximal action
        return [legal[i] for i in sorted(range(len(legal)), key=lambda i: -scores[i])]

    def placement_scores(self, view, seat):
        ranked = self._served_ranking(view, seat)[:self.candidates]
        world = setup_world(view)
        # streams from the public position, so the served RNG is untouched
        key = (view.setup_index, tuple(sorted(a['vertex'] for a in view.actions)), seat)
        base = random.Random(repr(key)).randrange(1 << 40)
        jobs = [(world, seat, action, base + r, self.max_turns) for action in ranked for r in range(self.playouts)]
        if self.workers > 1:
            with ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                bits = list(pool.map(playout, jobs, chunksize=4))
        else:
            bits = [playout(job) for job in jobs]
        n = self.playouts
        return [(action, sum(bits[i * n:(i + 1) * n])) for i, action in enumerate(ranked)]

    def choose(self, view, seat):
        if view.phase != 'setup_settlement' or len(view.actions) < 2:
            return super().choose(view, seat)
        scores = self.placement_scores(view, seat)
        best = max(range(len(scores)), key=lambda i: (scores[i][1], -i))
        return scores[best][0]
