"""Geometric mapping between the JSettlers classic (v1 encoding) board and the engine board.

JSettlers v1 coordinates are two hex digits ``0xAB`` along two diagonal axes; the
first runs north-west to south-east, the second south-west to north-east, and hexes
in a west-east row are 2 units apart (``doc/hexcoord.gif``). A hex centre is placed at
``A*u1 + B*u2`` with ``u1``/``u2`` unit vectors at -60/+60 degrees, which puts the six
neighbours of every hex at distance 2. A node's position is the centroid of the three
hexes (land or water) around it, as reported by ``SOCBoard.getAdjacentHexesToNode``.

The engine board (``Board.random``) is a pointy-top 19-hex layout. The mapping is the
one of the twelve hex-lattice symmetries (six rotations, optionally reflected) that
sends every JSettlers land-hex centre onto an engine tile centre; nodes and edges then
follow by position. Every element must map 1:1 or ``BoardMappingError`` is raised.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache

from catan_ai.engine import DESERT, Board, Resource

SQRT3 = math.sqrt(3)

# soc.game.SOCBoard hex types / SOCResourceConstants
TERRAIN = {1: Resource.BRICK, 2: Resource.ORE, 3: Resource.WOOL, 4: Resource.GRAIN,
           5: Resource.LUMBER, 6: DESERT}
TERRAIN_CODE = {v: k for k, v in TERRAIN.items()}
RESOURCE_CODE = {Resource.BRICK: 1, Resource.ORE: 2, Resource.WOOL: 3, Resource.GRAIN: 4,
                 Resource.LUMBER: 5}
CODE_RESOURCE = {v: k for k, v in RESOURCE_CODE.items()}
# soc.game.SOCBoard port types: MISC_PORT=0 (3:1), then one per resource code
PORT = {0: None, **{code: res for code, res in CODE_RESOURCE.items()}}
PORT_CODE = {None: 0, **RESOURCE_CODE}

STANDARD_TERRAIN = {Resource.BRICK: 3, Resource.LUMBER: 4, Resource.WOOL: 4, Resource.GRAIN: 4,
                    Resource.ORE: 3, DESERT: 1}
STANDARD_NUMBERS = sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
STANDARD_PORTS = {None: 4, **{r: 1 for r in Resource}}


class BoardMappingError(ValueError):
    pass


def js_point(coord: int) -> tuple[float, float]:
    """Cartesian position (engine scale: neighbour hexes sqrt(3) apart) of a v1 coordinate."""
    a, b = coord >> 4, coord & 0xF
    x, y = (a + b) / 2 - 7, (b - a) * SQRT3 / 2
    scale = SQRT3 / 2
    return x * scale, y * scale


def _transforms():
    for reflect in (False, True):
        for k in range(6):
            c, s = math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))

            def t(p, c=c, s=s, reflect=reflect):
                x, y = p
                if reflect:
                    y = -y
                return (c * x - s * y, s * x + c * y)
            yield (reflect, k), t


def _key(p, digits=3):
    return (round(p[0], digits) + 0.0, round(p[1], digits) + 0.0)


@lru_cache(maxsize=1)
def template() -> Board:
    """Engine board geometry (tile coordinates, vertex and edge indices)."""
    return Board.random(random.Random(0))


@dataclass(frozen=True)
class BoardMap:
    transform: tuple[bool, int]
    hex_to_tile: dict[int, int]
    node_to_vertex: dict[int, int]
    edge_to_engine: dict[int, tuple[int, int]]

    @property
    def tile_to_hex(self):
        return {v: k for k, v in self.hex_to_tile.items()}

    @property
    def vertex_to_node(self):
        return {v: k for k, v in self.node_to_vertex.items()}

    @property
    def engine_to_edge(self):
        return {v: k for k, v in self.edge_to_engine.items()}


def _layout_signature(js_board: dict):
    return (tuple(sorted(h['coord'] for h in js_board['hexes'])),
            tuple(sorted((n['coord'], tuple(sorted(n['hexes']))) for n in js_board['nodes'])),
            tuple(sorted((e['coord'], tuple(sorted(e['nodes']))) for e in js_board['edges'])))


def board_map(js_board: dict) -> BoardMap:
    return _board_map(_layout_signature(js_board))


@lru_cache(maxsize=8)
def _board_map(signature) -> BoardMap:
    hexes, nodes, edges = signature
    engine = template()
    tile_index = {_key((SQRT3 * (t.q + t.r / 2), 1.5 * t.r)): i for i, t in enumerate(engine.tiles)}
    vertex_index = {_key(p): i for i, p in enumerate(engine.vertices)}
    if len(hexes) != len(engine.tiles):
        raise BoardMappingError(f'expected {len(engine.tiles)} land hexes, got {len(hexes)}')
    for name, t in _transforms():
        hex_to_tile = {}
        for h in hexes:
            tile = tile_index.get(_key(t(js_point(h))))
            if tile is None:
                break
            hex_to_tile[h] = tile
        if len(hex_to_tile) != len(hexes) or len(set(hex_to_tile.values())) != len(hexes):
            continue
        node_to_vertex = {}
        for coord, around in nodes:
            if len(around) != 3:
                raise BoardMappingError(f'node {coord:#x} has {len(around)} adjacent hexes')
            pts = [js_point(h) for h in around]
            centroid = (sum(p[0] for p in pts) / 3, sum(p[1] for p in pts) / 3)
            vertex = vertex_index.get(_key(t(centroid)))
            if vertex is None:
                raise BoardMappingError(f'node {coord:#x} has no engine vertex')
            node_to_vertex[coord] = vertex
        if len(set(node_to_vertex.values())) != len(engine.vertices) or len(nodes) != len(engine.vertices):
            raise BoardMappingError('nodes do not map 1:1 onto engine vertices')
        engine_edges = set(engine.edges)
        edge_to_engine = {}
        for coord, (a, b) in edges:
            pair = tuple(sorted((node_to_vertex[a], node_to_vertex[b])))
            if pair not in engine_edges:
                raise BoardMappingError(f'edge {coord:#x} is not an engine edge')
            edge_to_engine[coord] = pair
        if len(set(edge_to_engine.values())) != len(engine_edges) or len(edges) != len(engine_edges):
            raise BoardMappingError('edges do not map 1:1 onto engine edges')
        return BoardMap(name, hex_to_tile, node_to_vertex, edge_to_engine)
    raise BoardMappingError('no lattice symmetry maps JSettlers land hexes onto engine tiles')


def engine_board(js_board: dict, *, check_standard: bool = True) -> tuple[Board, BoardMap]:
    """Engine Board with JSettlers' terrain, numbers and ports, plus the coordinate map."""
    m = board_map(js_board)
    base = template()
    by_tile = {}
    for h in js_board['hexes']:
        terrain = TERRAIN.get(h['type'])
        if terrain is None:
            raise BoardMappingError(f'unsupported hex type {h["type"]} at {h["coord"]:#x}')
        number = None if terrain == DESERT else h['number']
        if terrain != DESERT and number not in range(2, 13):
            raise BoardMappingError(f'bad number {number} at {h["coord"]:#x}')
        by_tile[m.hex_to_tile[h['coord']]] = (terrain, number)
    tiles = tuple(replace(t, resource=by_tile[i][0], number=by_tile[i][1]) for i, t in enumerate(base.tiles))
    ports = []
    for p in js_board['ports']:
        if p['type'] not in PORT:
            raise BoardMappingError(f'unsupported port type {p["type"]}')
        a, b = (m.node_to_vertex[n] for n in p['nodes'])
        edge = tuple(sorted((a, b)))
        if edge not in m.engine_to_edge or m.engine_to_edge[edge] != p['edge']:
            raise BoardMappingError(f'port edge {p["edge"]:#x} does not match its nodes')
        ports.append((edge, PORT[p['type']]))
    board = replace(base, tiles=tiles, ports=tuple(ports))
    if check_standard:
        check_base_board(board)
    return board, m


def check_base_board(board: Board) -> None:
    terrain = {}
    for t in board.tiles:
        terrain[t.resource] = terrain.get(t.resource, 0) + 1
    if terrain != STANDARD_TERRAIN:
        raise BoardMappingError(f'non-standard terrain mix {terrain}')
    numbers = sorted(t.number for t in board.tiles if t.number is not None)
    if numbers != STANDARD_NUMBERS:
        raise BoardMappingError(f'non-standard number tokens {numbers}')
    ports = {}
    for _, kind in board.ports:
        ports[kind] = ports.get(kind, 0) + 1
    if ports != STANDARD_PORTS:
        raise BoardMappingError(f'non-standard ports {ports}')
    coast = {e for e in board.edges if sum(set(e) <= set(t.vertices) for t in board.tiles) == 1}
    if any(edge not in coast for edge, _ in board.ports):
        raise BoardMappingError('port off the coast')


def js_board_from_engine(board: Board, m: BoardMap, js_layout: dict) -> dict:
    """Inverse direction: JSettlers-coordinate terrain/number/port facts from an engine board.

    ``js_layout`` supplies the coordinate lists (hexes/nodes/edges/ports positions).
    Used for the round-trip test.
    """
    tile_to_hex = m.tile_to_hex
    hexes = []
    for i, t in enumerate(board.tiles):
        hexes.append({'coord': tile_to_hex[i], 'type': TERRAIN_CODE[t.resource],
                      'number': t.number})
    v2n, e2c = m.vertex_to_node, m.engine_to_edge
    ports = [{'edge': e2c[tuple(edge)], 'type': PORT_CODE[kind],
              'nodes': sorted(v2n[v] for v in edge)} for edge, kind in board.ports]
    return {'hexes': sorted(hexes, key=lambda h: h['coord']),
            'ports': sorted(ports, key=lambda p: p['edge'])}
