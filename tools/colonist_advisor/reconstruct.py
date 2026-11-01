"""Turn a screen-read Position into an engine Game, and describe engine actions in screen terms.

Read-only: nothing here sends input to any application.
"""
import math
import random
from dataclasses import replace

from catan_ai.engine import Board, Game, Player, Resource, longest_road

from .schema import Position

ROW_LENGTHS = (3, 4, 5, 4, 3)
CORNER_ANGLES = {'top_right': -30, 'bottom_right': 30, 'bottom': 90, 'bottom_left': 150, 'top_left': -150, 'top': -90}
SIDE_ANGLES = {'right': 0, 'bottom_right': 60, 'bottom_left': 120, 'left': 180, 'top_left': -120, 'top_right': -60}
PHASES = {'setup': 'setup_settlement', 'roll': 'roll', 'actions': 'actions', 'move_robber': 'robber',
          'steal': 'steal', 'discard': 'discard', 'trade_response': 'trade_response'}
TEMPLATE = Board.random(random.Random(0))  # vertex and edge numbering do not depend on the terrain draw


def hex_to_axial(row, col):
    r = row - 2
    if not 0 <= col < ROW_LENGTHS[row]:
        raise ValueError(f'row {row} has no column {col}')
    return max(-2, -2 - r) + col, r


def tile_index(row, col):
    q, r = hex_to_axial(row, col)
    return next(i for i, t in enumerate(TEMPLATE.tiles) if (t.q, t.r) == (q, r))


def tile_name(index):
    tile = TEMPLATE.tiles[index]
    r = tile.r
    return tile.r + 2, tile.q - max(-2, -2 - r)


def _center(index):
    vs = TEMPLATE.tiles[index].vertices
    return (sum(TEMPLATE.vertices[v][0] for v in vs) / 6, sum(TEMPLATE.vertices[v][1] for v in vs) / 6)


def _angle(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def _nearest(angle, table):
    return min(table, key=lambda name: abs((angle - table[name] + 180) % 360 - 180))


def vertex_at(row, col, corner):
    index = tile_index(row, col)
    cx, cy = _center(index)
    return next(v for v in TEMPLATE.tiles[index].vertices
                if _nearest(_angle(TEMPLATE.vertices[v][0] - cx, TEMPLATE.vertices[v][1] - cy), CORNER_ANGLES) == corner)


def edge_at(row, col, side):
    index = tile_index(row, col)
    cx, cy = _center(index)
    vs = TEMPLATE.tiles[index].vertices
    for a, b in zip(vs, vs[1:] + vs[:1]):
        mx = (TEMPLATE.vertices[a][0] + TEMPLATE.vertices[b][0]) / 2 - cx
        my = (TEMPLATE.vertices[a][1] + TEMPLATE.vertices[b][1]) / 2 - cy
        if _nearest(_angle(mx, my), SIDE_ANGLES) == side:
            return tuple(sorted((a, b)))
    raise AssertionError('unreachable')


def check(position: Position) -> list[str]:
    """Consistency problems that mean the screen read is probably wrong."""
    problems = []
    cells = {(t.row, t.col) for t in position.tiles}
    if len(position.tiles) != 19 or len(cells) != 19:
        problems.append(f'expected 19 distinct hexes, read {len(cells)}')
    terrain = sorted(t.terrain for t in position.tiles)
    expected = sorted(['brick'] * 3 + ['lumber'] * 4 + ['wool'] * 4 + ['grain'] * 4 + ['ore'] * 3 + ['desert'])
    if terrain != expected:
        problems.append(f'terrain counts differ from the base game: {terrain}')
    numbers = sorted(t.number for t in position.tiles if t.number is not None)
    if numbers != [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]:
        problems.append(f'number tokens differ from the base game: {numbers}')
    if len(position.players) != 4:
        problems.append(f'the engine needs four players, read {len(position.players)}')
    if sum(p.is_me for p in position.players) != 1:
        problems.append('exactly one player must be marked as me')
    colors = {p.color for p in position.players}
    for item in [*position.buildings, *position.roads]:
        if item.color not in colors:
            problems.append(f'piece of unknown color {item.color}')
    if position.current_color not in colors:
        problems.append(f'current player color {position.current_color} is not a player')
    return problems


def to_game(position: Position) -> tuple[Game, int]:
    problems = check(position)
    if problems:
        raise ValueError('; '.join(problems))
    seat_of = {p.color: i for i, p in enumerate(position.players)}
    me = next(i for i, p in enumerate(position.players) if p.is_me)
    tiles = list(TEMPLATE.tiles)
    for t in position.tiles:
        i = tile_index(t.row, t.col)
        tiles[i] = replace(tiles[i], resource='desert' if t.terrain == 'desert' else Resource(t.terrain),
                           number=None if t.terrain == 'desert' else t.number)
    ports = tuple((edge_at(p.row, p.col, p.side), None if p.resource is None else Resource(p.resource))
                  for p in position.ports)
    board = Board(tuple(tiles), TEMPLATE.vertices, TEMPLATE.edges, TEMPLATE.neighbors, ports)
    players = [Player(p.name) for p in position.players]
    for b in position.buildings:
        target = players[seat_of[b.color]].cities if b.kind == 'city' else players[seat_of[b.color]].settlements
        target.add(vertex_at(b.row, b.col, b.corner))
    for road in position.roads:
        players[seat_of[road.color]].roads.add(edge_at(road.row, road.col, road.side))
    for seat, (player, read) in enumerate(zip(players, position.players)):
        player.knights = read.knights_played
        player.largest_army = read.largest_army
        player.longest_road_award = read.longest_road
        if seat == me:
            player.resources = {r: getattr(position.my_hand, r.value) for r in Resource}
            player.development = [(card, 0) for card in position.my_development_cards]
        else:
            # Hidden: only the counts are public. The engine view exposes no more than that.
            player.resources = {r: 0 for r in Resource}
            player.resources[Resource.BRICK] = read.resource_cards
            player.development = [('knight', -1)] * read.development_cards
    held = sum(len(p.development) for p in players) + sum(p.knights for p in players)
    holder = lambda flag: next((i for i, p in enumerate(position.players) if getattr(p, flag)), None)
    game = Game(board, players, random.Random(0), current=seat_of[position.current_color],
                phase=PHASES.get(position.phase, 'finished'), turn=20, rolled=position.dice,
                robber_tile=tile_index(position.robber.row, position.robber.col),
                deck=['knight'] * max(25 - held, 0), record_events=False)
    occupied = game.occupied()
    game.road_lengths = [longest_road(p.roads, {v for v, o in occupied.items() if o != s}) for s, p in enumerate(players)]
    game.longest_road_holder, game.largest_army_holder = holder('longest_road'), holder('largest_army')
    if game.phase == 'discard':
        game.discard_remaining = {me: sum(players[me].resources.values()) // 2}
        game.robber_actor = game.current
        game.current = me
    if game.phase == 'trade_response' and position.trade_offer is not None:
        offer = position.trade_offer
        bundle = lambda hand: {k: v for k, v in hand.model_dump().items() if v}
        game.trade_offer = {'owner': seat_of[offer.from_color], 'recipient': me,
                            'give': bundle(offer.gives_me), 'receive': bundle(offer.wants_from_me)}
        game.current = me
    return game, me


def from_game(game: Game, colors: list[str], me: int) -> Position:
    """The inverse, for tests: describe an engine Game as the screen reader would."""
    def corner_of(v):
        for i, t in enumerate(TEMPLATE.tiles):
            if v in t.vertices:
                row, col = tile_name(i)
                cx, cy = _center(i)
                return row, col, _nearest(_angle(TEMPLATE.vertices[v][0] - cx, TEMPLATE.vertices[v][1] - cy), CORNER_ANGLES)

    def side_of(edge):
        for i, t in enumerate(TEMPLATE.tiles):
            if set(edge) <= set(t.vertices):
                row, col = tile_name(i)
                for side in SIDE_ANGLES:
                    if edge_at(row, col, side) == tuple(sorted(edge)):
                        return row, col, side
    data = {
        'tiles': [{'row': tile_name(i)[0], 'col': tile_name(i)[1], 'terrain': getattr(t.resource, 'value', t.resource),
                   'number': t.number} for i, t in enumerate(game.board.tiles)],
        'robber': dict(zip(('row', 'col'), tile_name(game.robber_tile))),
        'ports': [dict(zip(('row', 'col', 'side'), side_of(edge)), resource=getattr(r, 'value', None))
                  for edge, r in game.board.ports],
        'buildings': [dict(zip(('row', 'col', 'corner'), corner_of(v)), color=colors[s], kind=kind)
                      for s, p in enumerate(game.players)
                      for kind, vs in (('settlement', p.settlements), ('city', p.cities)) for v in vs],
        'roads': [dict(zip(('row', 'col', 'side'), side_of(e)), color=colors[s])
                  for s, p in enumerate(game.players) for e in p.roads],
        'players': [{'color': colors[s], 'name': p.name, 'is_me': s == me, 'visible_points': p.public_points,
                     'resource_cards': sum(p.resources.values()), 'development_cards': len(p.development),
                     'knights_played': p.knights, 'longest_road': game.longest_road_holder == s,
                     'largest_army': game.largest_army_holder == s} for s, p in enumerate(game.players)],
        'my_hand': {r.value: n for r, n in game.players[me].resources.items()},
        'my_development_cards': [card for card, _ in game.players[me].development],
        'current_color': colors[game.current], 'phase': {v: k for k, v in PHASES.items()}.get(game.phase, 'other_player'),
        'dice': game.rolled, 'uncertain': [],
    }
    return Position.model_validate(data)


def describe_vertex(game, v):
    names = []
    for t in game.board.tiles:
        if v in t.vertices:
            names.append('desert' if t.number is None else f'{t.resource.value} {t.number}')
    return 'corner touching ' + ', '.join(names)


def describe(action, game, colors):
    kind = action['type']
    if kind in ('build_settlement', 'build_city'):
        return f"{kind.replace('_', ' ')} at the {describe_vertex(game, action['vertex'])}"
    if kind == 'build_road':
        a, b = action['edge']
        return f'build road between ({describe_vertex(game, a)}) and ({describe_vertex(game, b)})'
    if kind == 'move_robber':
        t = game.board.tiles[action['tile']]
        row, col = tile_name(action['tile'])
        label = 'desert' if t.number is None else f'{t.resource.value} {t.number}'
        return f'move robber to {label} (row {row + 1}, hex {col + 1})'
    if kind == 'steal':
        return f"steal from {colors[action['victim']]}"
    if kind == 'offer_trade':
        return f"offer {colors[action['recipient']]}: give {action['give']} for {action['receive']}"
    if kind == 'bank_trade':
        return f"bank/port trade: give {action['give']} for {action['receive']}"
    if kind == 'play_development':
        extra = {k: v for k, v in action.items() if k not in ('type', 'card')}
        return f"play {action['card'].replace('_', ' ')}" + (f' {extra}' if extra else '')
    return kind.replace('_', ' ')
