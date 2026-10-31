"""Differential action fixtures; full hands are test inputs, never policy inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from catan_ai.bots import make_policy
from catan_ai.catanatron_adapter import new_external_game, RESOURCE_TO_EXTERNAL
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import RESOURCES, ActionRecord, ActionPrompt
from catanatron.models.player import Color


def fixture(seed, phase='actions', bank_one=False):
    local = Game.new(['a', 'b', 'c', 'd'], seed)
    external, mapping = new_external_game(local.board, seed)
    bot = make_policy('tactical', seed)
    for _ in range(16):
        seat = local.current
        action = bot.choose(Observation.from_game(local, seat), seat)
        external.execute(mapping.translate_action(action, seat))
        local.apply(seat, action)
    local.phase, local.turn = phase, 8
    for s, player in enumerate(local.players):
        player.resources = {r: (4 if not bank_one else 18 if s == 0 else 0) for r in Resource}
        if bank_one and s == 1:
            player.resources = {r: 1 for r in Resource}
            player.resources[Resource.BRICK] = 0
        for r, count in player.resources.items():
            external.state.player_state[f'P{s}_{RESOURCE_TO_EXTERNAL[r]}_IN_HAND'] = count
    cards = ['knight', 'year_of_plenty', 'monopoly', 'road_building']
    local.players[0].development = [(c, 0) for c in cards]
    for c in cards:
        local.deck.remove(c)
        external.state.player_state[f'P0_{c.upper()}_IN_HAND'] = 1
        external.state.player_state[f'P0_{c.upper()}_OWNED_AT_START'] = True
    external.state.development_listdeck = [c.upper() for c in local.deck]
    external.state.resource_freqdeck = [local.bank[Resource(next(r for r, e in RESOURCE_TO_EXTERNAL.items() if e == name))]
                                        for name in RESOURCES]
    external.state.player_state['P0_HAS_ROLLED'] = phase == 'actions'
    external.state.current_prompt = ActionPrompt.PLAY_TURN
    external.playable_actions = generate_playable_actions(external.state)
    return local, external, mapping


def compare(local, external, mapping):
    assert local.current == external.state.current_player_index, 'current seat'
    for s, player in enumerate(local.players):
        color = list(Color)[s]
        for resource, count in player.resources.items():
            assert external.state.player_state[f'P{s}_{RESOURCE_TO_EXTERNAL[resource]}_IN_HAND'] == count, 'hand'
        assert external.state.player_state[f'P{s}_VICTORY_POINTS'] == player.public_points, 'points'
        for kind, vertices in [('SETTLEMENT', player.settlements), ('CITY', player.cities)]:
            assert set(external.state.buildings_by_color[color][kind]) == {mapping.vertices[v] for v in vertices}, kind
        assert {tuple(sorted(edge)) for edge in external.state.buildings_by_color[color]['ROAD']} == {
            tuple(sorted(mapping.vertices[v] for v in edge)) for edge in player.roads}, 'roads'
        for card in ['knight', 'year_of_plenty', 'monopoly', 'road_building', 'victory_point']:
            assert external.state.player_state[f'P{s}_{card.upper()}_IN_HAND'] == sum(c == card for c, _ in player.development), 'development'
    for resource, external_resource in RESOURCE_TO_EXTERNAL.items():
        assert external.state.resource_freqdeck[RESOURCES.index(external_resource)] == local.bank[Resource(resource)], 'bank'
    prompt = {'actions': ActionPrompt.PLAY_TURN, 'roll': ActionPrompt.PLAY_TURN,
              'free_roads': ActionPrompt.PLAY_TURN, 'robber': ActionPrompt.MOVE_ROBBER,
              'discard': ActionPrompt.DISCARD}
    assert external.state.current_prompt == prompt[local.phase], 'phase'
    if local.phase in {'actions', 'roll'}:
        assert external.state.player_state[f'P{local.current}_HAS_ROLLED'] == (local.phase == 'actions'), 'rolled flag'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    started = time.perf_counter()
    for seed in (1400040, 1400041):
        for phase in ('roll', 'actions'):
            for bank_one in (False, True):
                local, external, mapping = fixture(seed, phase, bank_one)
                rates = {r: local.trade_rate(0, r) for r in Resource}
                for action in local.legal_actions(0):
                    left, right = deepcopy(local), external.copy()
                    mapped = mapping.translate_action(action, 0, rates)
                    row = {'seed': seed, 'phase': phase, 'bank_one': bank_one, 'action': action}
                    try:
                        assert mapped in right.playable_actions, 'mapped action not legal'
                        left.apply(0, action)
                        record = None
                        if action['type'] == 'roll':
                            record = ActionRecord(mapped, (left.rolled // 2, left.rolled - left.rolled // 2))
                        elif action['type'] == 'buy_development':
                            record = ActionRecord(mapped, left.players[0].development[-1][0].upper())
                        right.execute(mapped, action_record=record)
                        compare(left, right, mapping)
                        row['pass'] = True
                    except (AssertionError, ValueError, KeyError) as exc:
                        row.update({'pass': False, 'error': str(exc)})
                    rows.append(row)
    result = {'scope': 'synthetic one-action transition fixtures', 'seconds': time.perf_counter() - started,
              'pass': all(r['pass'] for r in rows), 'cases': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    failures = [r for r in rows if not r['pass']]
    print(json.dumps({'cases': len(rows), 'failures': len(failures), 'examples': failures[:8]}, indent=2))


if __name__ == '__main__':
    main()
