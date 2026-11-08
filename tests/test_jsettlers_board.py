"""JSettlers classic board <-> engine board mapping round trip on JSettlers-generated boards."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'jsettlers'))

from catan_ai.engine import DESERT  # noqa: E402
from jsbridge.board import (BoardMappingError, PORT_CODE, TERRAIN_CODE, engine_board,  # noqa: E402
                            js_board_from_engine)

BOARDS = json.loads((ROOT / 'tools' / 'jsettlers' / 'fixtures' / 'boards.json').read_text())


def test_fixture_has_enough_generated_boards():
    assert len(BOARDS) >= 50
    # Different boards, not one board repeated.
    assert len({json.dumps(b['hexes'], sort_keys=True) for b in BOARDS}) == len(BOARDS)


@pytest.mark.parametrize('index', range(len(BOARDS)))
def test_board_round_trip(index):
    js = BOARDS[index]
    board, m = engine_board(js)
    # Hexes: 19 land hexes map 1:1, terrain and number preserved both ways.
    assert sorted(m.hex_to_tile.values()) == list(range(19))
    for h in js['hexes']:
        tile = board.tiles[m.hex_to_tile[h['coord']]]
        assert TERRAIN_CODE[tile.resource] == h['type']
        assert tile.number == (None if tile.resource == DESERT else h['number'])
    back = js_board_from_engine(board, m, js)
    assert [(h['coord'], h['type'], h['number'] or 0) for h in back['hexes']] == \
        sorted((h['coord'], h['type'], h['number'] if h['type'] != 6 else 0) for h in js['hexes'])
    # Nodes and edges: bijections, hex corners and adjacency preserved.
    assert sorted(m.node_to_vertex.values()) == list(range(len(board.vertices)))
    assert sorted(m.edge_to_engine.values()) == sorted(board.edges)
    for entry in js['hex_nodes']:
        tile = board.tiles[m.hex_to_tile[entry['hex']]]
        assert {m.node_to_vertex[n] for n in entry['nodes']} == set(tile.vertices)
    for e in js['edges']:
        a, b = (m.node_to_vertex[n] for n in e['nodes'])
        assert b in board.neighbors[a] and a in board.neighbors[b]
        assert m.engine_to_edge[m.edge_to_engine[e['coord']]] == e['coord']
    for coord, vertex in m.node_to_vertex.items():
        assert m.vertex_to_node[vertex] == coord
    # Ports: same edges and types both ways.
    assert len(board.ports) == 9
    assert back['ports'] == sorted(({'edge': p['edge'], 'type': p['type'], 'nodes': sorted(p['nodes'])}
                                    for p in js['ports']), key=lambda p: p['edge'])
    for p in js['ports']:
        assert p['type'] == p['node_type'] == p['node_type_b']
        edge = m.edge_to_engine[p['edge']]
        assert (edge, next(k for k, c in PORT_CODE.items() if c == p['type'])) in board.ports
    # The robber starts on the desert.
    assert board.tiles[m.hex_to_tile[js['robber']]].resource == DESERT


def test_rejects_non_standard_board():
    js = json.loads(json.dumps(BOARDS[0]))
    js['hexes'][0]['type'] = 1 if js['hexes'][0]['type'] != 1 else 2
    with pytest.raises(BoardMappingError):
        engine_board(js)
