"""Check exact board and setup parity using the external search implementation.

Run in the separately installed, pinned Catanatron environment with PYTHONPATH=src.
This diagnostic uses only setup positions, whose board and holdings are public.
It is not a full-game comparator or strength evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from catan_ai.catanatron_adapter import new_external_game, RESOURCE_TO_EXTERNAL
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer

PIN = 'ecf931181b9a65bb4116a2153fb78c16f1438e00'


def verify(seed):
    local = Game.new(['a', 'b', 'c', 'd'], seed)
    external, mapping = new_external_game(local.board, seed)
    colors = list(Color)
    bots = [AlphaBetaPlayer(color) for color in colors]
    assert len(mapping.vertices) == 54 and len(mapping.tiles) == 19
    assert len(mapping.external_map.ports_by_id) == 9
    for index, tile in enumerate(local.board.tiles):
        other = mapping.external_map.land_tiles[mapping.tiles[index]]
        assert other.number == tile.number
        assert other.resource == RESOURCE_TO_EXTERNAL.get(tile.resource)
        assert set(other.nodes.values()) == {mapping.vertices[v] for v in tile.vertices}
    expected_ports = {}
    for edge, resource in local.board.ports:
        expected_ports.setdefault(RESOURCE_TO_EXTERNAL.get(resource), set()).update(mapping.vertices[v] for v in edge)
    assert dict(mapping.external_map.port_nodes) == expected_ports
    trace = []
    for step in range(16):
        seat = local.current
        assert external.state.current_player_index == seat
        legal = local.legal_actions(seat)
        assert {mapping.translate_action(a, seat) for a in legal} == set(external.playable_actions)
        started = time.perf_counter()
        action = bots[seat].decide(external, external.playable_actions)
        seconds = time.perf_counter() - started
        selected = mapping.local_action(action, legal, seat)
        local.apply(seat, selected)
        external.execute(action)
        for s, player in enumerate(local.players):
            for resource, count in player.resources.items():
                assert external.state.player_state[f'P{s}_{RESOURCE_TO_EXTERNAL[resource]}_IN_HAND'] == count
            assert external.state.player_state[f'P{s}_VICTORY_POINTS'] == player.public_points
            for resource in player.resources:
                ports = external.state.board.get_player_port_resources(colors[s])
                rate = 2 if RESOURCE_TO_EXTERNAL[resource] in ports else 3 if None in ports else 4
                assert rate == local.trade_rate(s, resource)
        trace.append({'step': step, 'seat': seat, 'action': selected, 'seconds': seconds})
    assert local.phase == 'roll' and not external.state.is_initial_build_phase
    assert local.current == external.state.current_player_index == 0
    return {'seed': seed, 'steps': trace, 'pass': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, default=Path('runs/external/catanatron'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sha = subprocess.check_output(['git', '-C', str(args.checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != PIN:
        raise RuntimeError('unexpected external revision')
    # Verify installed search bytes as well as the checkout pointer.
    import catanatron.players.minimax as minimax
    installed = Path(minimax.__file__)
    source = args.checkout / 'catanatron/catanatron/players/minimax.py'
    assert installed.read_bytes() == source.read_bytes()
    started = time.perf_counter()
    rows = [verify(seed) for seed in range(1400030, 1400038)]
    result = {'scope': 'public board and complete setup only', 'full_comparator_verified': False,
              'external_commit': sha, 'external_search_sha256': hashlib.sha256(installed.read_bytes()).hexdigest(),
              'source': source_files(), 'host': platform.node(), 'process_cap': 1,
              'seconds': time.perf_counter() - started, 'games': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'boards': len(rows), 'setup_decisions': sum(len(r['steps']) for r in rows),
                      'seconds': result['seconds'], 'full_comparator_verified': False}))


if __name__ == '__main__':
    main()
