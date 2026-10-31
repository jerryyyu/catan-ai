"""Geometry and action translation for the optional pinned external comparator.

The external package is imported only when the adapter is used. Geometry is
public information; these helpers never receive opponent hands or engine RNG.
"""
from dataclasses import dataclass
import random


RESOURCE_TO_EXTERNAL = {'brick': 'BRICK', 'lumber': 'WOOD', 'wool': 'SHEEP',
                        'grain': 'WHEAT', 'ore': 'ORE'}


@dataclass
class BoardMapping:
    external_map: object
    vertices: dict[int, int]
    tiles: dict[int, tuple[int, int, int]]

    def translate_action(self, action, seat, rates=None):
        from catanatron.models.enums import Action, ActionType, RESOURCES
        from catanatron.models.player import Color
        kind = action['type']
        simple = {'roll': 'ROLL', 'end_turn': 'END_TURN',
                  'buy_development': 'BUY_DEVELOPMENT_CARD'}
        builds = {'build_settlement': 'BUILD_SETTLEMENT', 'build_city': 'BUILD_CITY',
                  'build_road': 'BUILD_ROAD'}
        value = None
        if kind in simple:
            external_kind = simple[kind]
        elif kind in builds:
            external_kind = builds[kind]
            value = (tuple(sorted(self.vertices[v] for v in action['edge']))
                     if kind == 'build_road' else self.vertices[action['vertex']])
        elif kind == 'discard':
            external_kind = 'DISCARD_RESOURCE'
            value = RESOURCE_TO_EXTERNAL[action['resource']]
        elif kind == 'bank_trade':
            if rates is None:
                raise ValueError('bank trade translation requires observed trade rates')
            external_kind = 'MARITIME_TRADE'
            rate = rates[action['give']]
            value = tuple([RESOURCE_TO_EXTERNAL[action['give']]] * rate
                          + [None] * (4 - rate) + [RESOURCE_TO_EXTERNAL[action['receive']]])
        elif kind == 'play_development':
            external_kind = {'knight': 'PLAY_KNIGHT_CARD', 'monopoly': 'PLAY_MONOPOLY',
                             'year_of_plenty': 'PLAY_YEAR_OF_PLENTY',
                             'road_building': 'PLAY_ROAD_BUILDING'}[action['card']]
            if action['card'] == 'monopoly':
                value = RESOURCE_TO_EXTERNAL[action['resource']]
            elif action['card'] == 'year_of_plenty':
                value = tuple(sorted((RESOURCE_TO_EXTERNAL[r] for r in action['resources']),
                                     key=RESOURCES.index))
        else:
            raise NotImplementedError(f'external translation not verified for {kind}')
        return Action(list(Color)[seat], ActionType[external_kind], value)

    def local_action(self, external, legal_actions, seat, rates=None):
        for action in legal_actions:
            try:
                if self.translate_action(action, seat, rates) == external:
                    return action
            except NotImplementedError:
                continue
        raise ValueError('external decision has no matching local legal action')


def map_board(board):
    """Map the full public geometry, including exact port edges, not seed values."""
    import networkx as nx
    from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, PORT_DIRECTION_TO_NODEREFS
    from catanatron.models.tiles import LandTile, Port, Water

    template = CatanMap.from_template(BASE_MAP_TEMPLATE, rng=random.Random(0))
    local_graph, external_graph = nx.Graph(), nx.Graph()
    local_graph.add_edges_from(board.edges)
    for tile in template.land_tiles.values():
        external_graph.add_edges_from(tile.edges.values())
    matcher = nx.algorithms.isomorphism.GraphMatcher(local_graph, external_graph)
    if not matcher.is_isomorphic():
        raise ValueError('external board topology differs from local board')
    vertices = dict(matcher.mapping)
    land_by_nodes = {frozenset(t.nodes.values()): (coord, t)
                     for coord, t in template.land_tiles.items()}
    tiles = {}
    mapped_tiles = {coord: Water(tile.nodes, tile.edges)
                    for coord, tile in template.tiles.items() if not isinstance(tile, LandTile)}
    for index, tile in enumerate(board.tiles):
        coord, target = land_by_nodes[frozenset(vertices[v] for v in tile.vertices)]
        resource = None if tile.resource == 'desert' else RESOURCE_TO_EXTERNAL[tile.resource]
        mapped_tiles[coord] = LandTile(target.id, resource, tile.number, target.nodes, target.edges)
        tiles[index] = coord
    for index, (edge, resource) in enumerate(board.ports):
        target_edge = {vertices[v] for v in edge}
        matches = [(coord, tile, direction)
                   for coord, tile in mapped_tiles.items() if isinstance(tile, Water)
                   for direction, (a, b) in PORT_DIRECTION_TO_NODEREFS.items()
                   if {tile.nodes[a], tile.nodes[b]} == target_edge]
        if len(matches) != 1:
            raise ValueError('port edge does not identify a unique external coast location')
        coord, tile, direction = matches[0]
        mapped_tiles[coord] = Port(index, None if resource is None else RESOURCE_TO_EXTERNAL[resource],
                                   direction, tile.nodes, tile.edges)
    return BoardMapping(CatanMap.from_tiles(mapped_tiles), vertices, tiles)


def new_external_game(board, seed):
    """Create an empty external arena with identical public board and seat order."""
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer
    from catanatron.models.actions import generate_playable_actions
    mapping = map_board(board)
    players = [RandomPlayer(color) for color in Color]
    game = Game(players, seed=seed, catan_map=mapping.external_map)
    game.state.players = players
    game.state.colors = tuple(Color)
    game.state.color_to_index = {color: index for index, color in enumerate(Color)}
    game.playable_actions = generate_playable_actions(game.state)
    return game, mapping


def from_observation(view, belief, seed):
    """Construct a search world solely from the acting player's information.

    Unsupported protocols fail explicitly until their transitions are verified.
    The belief sampler owns opponent-card reconstruction; no real game is input.
    """
    from .observation import Observation
    from .external_compatibility import install_rule_compatibility
    install_rule_compatibility()
    if not isinstance(view, Observation):
        raise TypeError('external policy input must be a detached Observation')
    if view.phase not in {'actions', 'roll', 'free_roads', 'setup_settlement', 'setup_road',
                          'discard', 'robber', 'steal', 'trade_response'}:
        raise NotImplementedError(f'external state translation not verified for {view.phase}')
    world = belief.sample(view, seed)
    external, mapping = new_external_game(view.board, seed)
    import networkx as nx
    from catanatron.models.actions import generate_playable_actions
    from catanatron.models.enums import ActionPrompt, RESOURCES, Action, ActionType
    from catanatron.models.player import Color
    state, board = external.state, external.state.board
    colors = list(Color)
    for seat, player in enumerate(world.players):
        color = colors[seat]
        for kind, vertices in [('SETTLEMENT', player.settlements), ('CITY', player.cities)]:
            locations = sorted(mapping.vertices[v] for v in vertices)
            state.buildings_by_color[color][kind] = locations
            for vertex in locations:
                board.buildings[vertex] = (color, kind)
                board.board_buildable_ids.discard(vertex)
                for neighbor in board.buildable_subgraph.neighbors(vertex):
                    board.board_buildable_ids.discard(neighbor)
        edges = [tuple(sorted(mapping.vertices[v] for v in edge)) for edge in sorted(player.roads)]
        state.buildings_by_color[color]['ROAD'] = edges
        for a, b in edges:
            board.roads[a, b] = board.roads[b, a] = color
        values = state.player_state
        for resource, count in player.resources.items():
            values[f'P{seat}_{RESOURCE_TO_EXTERNAL[resource]}_IN_HAND'] = count
        for card in ('knight', 'year_of_plenty', 'monopoly', 'road_building', 'victory_point'):
            values[f'P{seat}_{card.upper()}_IN_HAND'] = sum(c == card for c, _ in player.development)
            if card != 'victory_point':
                values[f'P{seat}_{card.upper()}_OWNED_AT_START'] = any(
                    c == card and bought < world.turn for c, bought in player.development)
        values[f'P{seat}_PLAYED_KNIGHT'] = player.knights
        values[f'P{seat}_VICTORY_POINTS'] = player.public_points
        values[f'P{seat}_ACTUAL_VICTORY_POINTS'] = player.points
        values[f'P{seat}_SETTLEMENTS_AVAILABLE'] = 5 - len(player.settlements)
        values[f'P{seat}_CITIES_AVAILABLE'] = 4 - len(player.cities)
        values[f'P{seat}_ROADS_AVAILABLE'] = 15 - len(player.roads)
        values[f'P{seat}_HAS_ROAD'] = seat == world.longest_road_holder
        values[f'P{seat}_HAS_ARMY'] = seat == world.largest_army_holder
        values[f'P{seat}_LONGEST_ROAD_LENGTH'] = world.road_lengths[seat]
        values[f'P{seat}_HAS_ROLLED'] = seat == view.seat and (
            view.phase == 'actions' or view.phase in {'free_roads', 'robber', 'steal'}
            and view.return_phase == 'actions')
        values[f'P{seat}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'] = seat == view.seat and view.played_development
        board.road_lengths[color] = world.road_lengths[seat]
    for color in colors:
        graph = nx.Graph()
        graph.add_edges_from(state.buildings_by_color[color]['ROAD'])
        graph.add_nodes_from(state.buildings_by_color[color]['SETTLEMENT'] + state.buildings_by_color[color]['CITY'])
        graph.remove_nodes_from([v for v, (owner, _) in board.buildings.items() if owner != color])
        board.connected_components[color] = [set(nodes) for nodes in nx.connected_components(graph)]
    board.road_color = None if world.longest_road_holder is None else colors[world.longest_road_holder]
    board.road_length = 0 if world.longest_road_holder is None else world.road_lengths[world.longest_road_holder]
    board.robber_coordinate = mapping.tiles[view.robber_tile]
    reverse_resources = {external: local for local, external in RESOURCE_TO_EXTERNAL.items()}
    state.resource_freqdeck = [world.bank[reverse_resources[r]] for r in RESOURCES]
    state.development_listdeck = [card.upper() for card in world.deck]
    state.current_player_index = view.seat
    state.current_turn_index = (view.robber_actor if view.phase == 'discard' else
                                view.trade_offer['owner'] if view.phase == 'trade_response' else view.seat)
    state.num_turns = view.turn
    state.is_initial_build_phase = view.phase.startswith('setup')
    state.current_prompt = {'setup_settlement': ActionPrompt.BUILD_INITIAL_SETTLEMENT,
                           'setup_road': ActionPrompt.BUILD_INITIAL_ROAD,
                           'discard': ActionPrompt.DISCARD,
                           'robber': ActionPrompt.MOVE_ROBBER,
                           'steal': ActionPrompt.MOVE_ROBBER}.get(view.phase, ActionPrompt.PLAY_TURN)
    state.is_discarding = view.phase == 'discard'
    state.discard_counts = [view.discard_remaining.get(s, 0) for s in range(4)]
    if view.phase == 'discard':
        state.player_state[f'P{view.robber_actor}_HAS_ROLLED'] = True
    if view.phase == 'setup_road':
        locations = state.buildings_by_color[colors[view.seat]]['SETTLEMENT']
        pending = mapping.vertices[view.pending_setup_vertex]
        locations.remove(pending)
        locations.append(pending)
    state.is_road_building = view.phase == 'free_roads'
    state.free_roads_available = view.free_roads
    external.playable_actions = generate_playable_actions(state)
    if view.phase == 'trade_response':
        external.playable_actions = []
    elif view.phase == 'steal':
        external.playable_actions = [Action(colors[view.seat], ActionType.MOVE_ROBBER,
                                           (mapping.tiles[view.robber_tile], colors[a['victim']]))
                                     for a in view.actions]
    elif view.phase == 'robber':
        if {a.value[0] for a in external.playable_actions} != {mapping.tiles[a['tile']] for a in view.actions}:
            raise ValueError('external robber locations differ from observation')
    else:
        expected = {mapping.translate_action(action, view.seat, view.rates) for action in view.actions}
        if expected != set(external.playable_actions):
            raise ValueError('external root legal actions differ from observation')
    return external, mapping


def decision_to_local(action, view, mapping):
    """Translate the external combined robber decision into our current phase."""
    from catanatron.models.player import Color
    if view.phase == 'robber':
        inverse = {coord: index for index, coord in mapping.tiles.items()}
        result = {'type': 'move_robber', 'tile': inverse[action.value[0]]}
    elif view.phase == 'steal':
        result = {'type': 'steal', 'victim': list(Color).index(action.value[1])}
    else:
        result = mapping.local_action(action, view.actions, view.seat, view.rates)
    if result not in view.actions:
        raise ValueError('external decision is not a legal observed action')
    return result


def choose_trade_response(external, view):
    """Use the external leaf evaluator for our atomic bilateral trade protocol.

    Stock AlphaBeta's successor enumerator does not implement trade responses.
    This explicit one-ply adaptation evaluates the same accept/reject outcomes
    as our engine; it does not simulate the external multi-recipient auction.
    """
    from catanatron.models.enums import Action, ActionType, RESOURCES
    from catanatron.models.player import Color
    from catanatron.players.value import get_value_fn, DEFAULT_WEIGHTS
    reject = {'type': 'reject_trade'}
    if {'type': 'accept_trade'} not in view.actions:
        return reject
    evaluator = get_value_fn('base_fn', DEFAULT_WEIGHTS)
    colors = list(Color)
    before = evaluator(external, colors[view.seat])
    offer = view.trade_offer
    inverse = {external: local for local, external in RESOURCE_TO_EXTERNAL.items()}
    value = tuple(offer[side].get(inverse[r], 0) for side in ('give', 'receive') for r in RESOURCES)
    after = external.copy()
    after.execute(Action(colors[offer['owner']], ActionType.CONFIRM_TRADE,
                         value + (colors[view.seat],)), validate_action=False)
    return {'type': 'accept_trade'} if evaluator(after, colors[view.seat]) > before else reject
