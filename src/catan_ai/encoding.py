"""Observation encoder v2 for learned value models (model audit, 2026-11-04).

Replaces the 40 hand-made aggregates with the full public picture from the
seat's point of view, on the fixed board topology (19 tiles, 54 vertices, 72
edges; only terrain, numbers and port types are shuffled per game):

- board: per tile resource one-hot (6), pips/5, robber flag; per vertex a
  seat-relative settlement/city flag for each player (8) and its port type (6);
  per edge a seat-relative road owner (4);
- per player, in seat-relative order (me, next, ...): public points, settlements,
  cities, roads, longest-road length, award flags, knights played, development
  cards held, resource cards held, hand per resource (own exact, opponents from
  public card counting), robber-adjusted income per resource, ports (generic and
  each 2:1), open settlement sites;
- own development cards by type, own trade rates; turn, absolute seat (setup
  order), and the bank estimated from public counts.

Uses only the seat's observation and the card-counted world built from it:
hidden opponent hands enter only as public-history estimates, never directly.
"""
import math

from .engine import Resource
from .scoring import longest_road

RESOURCES = tuple(Resource)
TERRAIN = RESOURCES + ('desert',)
CARDS = ('victory_point', 'knight', 'monopoly', 'year_of_plenty', 'road_building')
PORTS = (None,) + RESOURCES


def _pips(number):
    return 0 if number is None else 6 - abs(7 - number)


def encode(view, world):
    """Feature vector for ``view.seat``; ``world`` is the planner's card-counted world for this view."""
    seat = view.seat
    order = [(seat + k) % 4 for k in range(4)]
    rel = {s: k for k, s in enumerate(order)}
    board = view.board
    x = []
    # tiles
    for i, tile in enumerate(board.tiles):
        kind = tile.resource if isinstance(tile.resource, Resource) else 'desert'
        x += [float(kind == t) for t in TERRAIN]
        x += [_pips(tile.number) / 5, float(i == view.robber_tile)]
    # vertices: ownership and ports
    owner = {}
    for s, p in enumerate(view.players):
        for v in p.settlements:
            owner[v] = (rel[s], 0)
        for v in p.cities:
            owner[v] = (rel[s], 1)
    port_at = {}
    for vertices, kind in board.ports:
        for v in vertices:
            port_at[v] = kind
    for v in range(len(board.vertices)):
        flags = [0.] * 8
        if v in owner:
            k, city = owner[v]
            flags[2 * k + city] = 1.
        x += flags
        kind = port_at.get(v, 'none')
        x += [float(kind == p) if kind != 'none' else 0. for p in PORTS]
    # edges
    road_owner = {}
    for s, p in enumerate(view.players):
        for edge in p.roads:
            road_owner[tuple(sorted(edge))] = rel[s]
    for edge in board.edges:
        k = road_owner.get(tuple(sorted(edge)))
        x += [float(k == j) for j in range(4)]
    # per player
    occupied = world.occupied()
    for s in order:
        p, wp = view.players[s], world.players[s]
        blocked = {v for v, o in occupied.items() if o != s}
        hand = p.resources if p.resources is not None else {r: wp.resources[r] for r in RESOURCES}
        income = {r: 0. for r in RESOURCES}
        for i, tile in enumerate(board.tiles):
            if i == view.robber_tile or tile.number is None or not isinstance(tile.resource, Resource):
                continue
            n = sum((v in p.settlements) + 2 * (v in p.cities) for v in tile.vertices)
            income[tile.resource] += n * _pips(tile.number)
        rates = world.trade_rates(s)
        x += [p.public_points / 10, len(p.settlements) / 5, len(p.cities) / 4, len(p.roads) / 15,
              longest_road(set(p.roads), blocked) / 15, float(view.longest_road_holder == s),
              float(view.largest_army_holder == s), p.knights / 14, p.development_count / 25,
              p.resource_count / 19]
        x += [hand[r] / 19 for r in RESOURCES]
        x += [income[r] / 30 for r in RESOURCES]
        x += [float(3 in rates.values())] + [float(rates[r] == 2) for r in RESOURCES]
        x += [len(world._legal_settlements(s)) / 10]
    # own cards and rates, globals
    own = view.players[seat]
    x += [sum(card == c for card in (own.development or ())) for c in CARDS]
    x += [view.rates[r] / 4 for r in RESOURCES]
    x += [min(view.turn, 400) / 400] + [float(seat == j) for j in range(4)]
    estimated = {r: sum((view.players[s].resources or world.players[s].resources)[r] for s in range(4))
                 for r in RESOURCES}
    x += [max(0., 19 - estimated[r]) / 19 for r in RESOURCES]
    return x


def size():
    return 19 * 8 + 54 * 14 + 72 * 4 + 4 * 27 + 5 + 5 + 1 + 4 + 5


# ----------------------------------------------------------- board symmetries
TILE_BLOCK, VERTEX_BLOCK, EDGE_BLOCK = 8, 14, 4


def symmetry_permutations(board):
    """The 12 symmetries of the hex board (6 rotations x optional reflection) as index
    permutations of the encoding: ``x_sym = [x[p[i]] for i in range(len(x))]``.

    The board topology is identical in every game (only contents are shuffled) and
    centred on the origin, so each symmetry maps tiles, vertices and edges to tiles,
    vertices and edges. Only the board block moves; per-player and global features
    are orientation-free. Used for training-time augmentation (AlphaGo-style).
    """
    def key(x, y):
        return (round(x, 3) + 0., round(y, 3) + 0.)
    vindex = {key(x, y): i for i, (x, y) in enumerate(board.vertices)}
    centres = [(sum(board.vertices[v][0] for v in t.vertices) / len(t.vertices),
                sum(board.vertices[v][1] for v in t.vertices) / len(t.vertices)) for t in board.tiles]
    tindex = {key(x, y): i for i, (x, y) in enumerate(centres)}
    eindex = {tuple(sorted(e)): i for i, e in enumerate(board.edges)}
    n_tiles, n_vertices = len(board.tiles), len(board.vertices)
    v0 = n_tiles * TILE_BLOCK
    e0 = v0 + n_vertices * VERTEX_BLOCK
    total = size()
    perms = []
    for reflect in (False, True):
        for k in range(6):
            c, s = math.cos(math.pi / 3 * k), math.sin(math.pi / 3 * k)

            def move(x, y, reflect=reflect, c=c, s=s):
                if reflect:
                    y = -y
                return key(c * x - s * y, s * x + c * y)
            vmap = [vindex[move(*board.vertices[v])] for v in range(n_vertices)]
            tmap = [tindex[move(*centres[t])] for t in range(n_tiles)]
            emap = [eindex[tuple(sorted((vmap[a], vmap[b])))] for a, b in board.edges]
            perm = list(range(total))
            # x_sym[new position] = x[old position]: new index of old item i is map[i]
            for t in range(n_tiles):
                for j in range(TILE_BLOCK):
                    perm[tmap[t] * TILE_BLOCK + j] = t * TILE_BLOCK + j
            for v in range(n_vertices):
                for j in range(VERTEX_BLOCK):
                    perm[v0 + vmap[v] * VERTEX_BLOCK + j] = v0 + v * VERTEX_BLOCK + j
            for e in range(len(board.edges)):
                for j in range(EDGE_BLOCK):
                    perm[e0 + emap[e] * EDGE_BLOCK + j] = e0 + e * EDGE_BLOCK + j
            perms.append(perm)
    return perms
