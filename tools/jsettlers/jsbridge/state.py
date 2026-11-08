"""Reconstruct an engine Game and Observation from a JSettlers bot-side snapshot.

The snapshot (built by ``catanai.jsettlers.StateJson``) holds only what the JSettlers
robot client knows as that seat: the public board and pieces, every player's
resource total, the resources JSettlers has seen each opponent gain or spend (its
known amounts) plus the count it could not see (``unknown``), development-card
counts, and our own exact hand and cards.

Seats are rotated so JSettlers' first player is engine seat 0; the engine's fixed
setup order ``0,1,2,3,3,2,1,0`` and ``current = (current + 1) % 4`` then match
JSettlers' clockwise order.

Opponent hands. The served planner reads opponents' hands only through
``card_planner.resource_estimates(view.events)``. The bridge supplies a short
synthetic history whose fold equals JSettlers' own bot-side tracking: one event
with ``counts`` all zero per public trade answer and per development-card purchase
(they leave every estimate at zero), then a single ``jsettlers_hands`` event whose
``resource_deltas`` give each opponent its known amounts plus its unknown cards
spread evenly over the five resources, and whose ``counts`` are the true totals.
The fold therefore returns exactly ``known + unknown / 5`` per resource. Opponent
hands inside the engine Game are placeholders with the right totals; the
Observation never exposes them.
"""
from __future__ import annotations

import random
from dataclasses import replace

from catan_ai.engine import RESOURCES, Game, Player, Resource
from catan_ai.observation import Observation
from catan_ai.scoring import longest_road

from .board import CODE_RESOURCE, engine_board

# soc.game.SOCDevCardConstants
DEV_CARDS = {9: 'knight', 1: 'road_building', 2: 'year_of_plenty', 3: 'monopoly',
             4: 'victory_point', 5: 'victory_point', 6: 'victory_point', 7: 'victory_point',
             8: 'victory_point'}
DEV_CODES = {'knight': 9, 'road_building': 1, 'year_of_plenty': 2, 'monopoly': 3}

DECISIONS = ('setup_settlement', 'setup_road', 'roll', 'actions', 'free_roads', 'robber',
             'steal', 'discard', 'trade_response')

# Keys a snapshot may carry for a seat other than the bridge's own.
OPPONENT_KEYS = frozenset({'pn', 'settlements', 'cities', 'roads', 'known', 'unknown', 'total',
                           'dev_count', 'knights', 'public_vp', 'played_dev'})
OWN_KEYS = frozenset({'resources', 'dev'})


class SnapshotError(ValueError):
    pass


class Seats:
    """JSettlers player number <-> engine seat, rotated so the first player is seat 0."""

    def __init__(self, first: int):
        if first not in range(4):
            raise SnapshotError(f'bad first player {first}')
        self.first = first

    def engine(self, pn: int) -> int:
        return (pn - self.first) % 4

    def js(self, seat: int) -> int:
        return (seat + self.first) % 4


def _hand(codes: dict) -> dict[Resource, int]:
    hand = {r: 0 for r in RESOURCES}
    for code, n in (codes or {}).items():
        hand[CODE_RESOURCE[int(code)]] += int(n)
    return hand


def engine_turn(snap: dict) -> int:
    return max(int(snap.get('turn_count', 0)) - 1, 0)


def check_hidden_boundary(snap: dict) -> None:
    """Raise if the snapshot carries opponent information JSettlers does not show the seat."""
    me = snap['seat']
    for p in snap['players']:
        extra = set(p) - OPPONENT_KEYS
        if extra:
            raise SnapshotError(f'player {p["pn"]} carries non-public keys {sorted(extra)}')
        known = sum(int(n) for n in p['known'].values())
        if known + int(p['unknown']) != int(p['total']):
            raise SnapshotError(f'player {p["pn"]} known+unknown != total')
    own = snap['own']
    if set(own) - OWN_KEYS:
        raise SnapshotError('own block carries unexpected keys')
    mine = next(p for p in snap['players'] if p['pn'] == me)
    if mine['unknown'] != 0 or _hand(mine['known']) != _hand(own['resources']):
        raise SnapshotError('own hand is not fully known')


def synthetic_events(snap: dict, seats: Seats, turn: int) -> tuple[dict, ...]:
    zero = [0, 0, 0, 0]
    events = []

    def event(seat, action, **extra):
        events.append({'seat': seat, 'turn': turn, 'phase': 'jsettlers', 'action': action,
                       'counts': list(extra.pop('counts', zero)), 'public_points': [0] * 4,
                       'winner': None, 'turn_owner': seat, **extra})

    for pn, (accepted, answered) in enumerate(snap.get('trade_answers', [[0, 0]] * 4)):
        for i in range(int(answered)):
            event(seats.engine(pn), {'type': 'accept_trade' if i < int(accepted) else 'reject_trade'})
    for _ in range(max(25 - int(snap.get('dev_remaining', 25)), 0)):
        event(seats.engine(snap['current']), {'type': 'buy_development'})
    deltas, counts = [None] * 4, [0] * 4
    for p in snap['players']:
        seat = seats.engine(p['pn'])
        known = _hand(p['known'])
        unknown = int(p['unknown'])
        deltas[seat] = {r.value: known[r] + unknown / len(RESOURCES) for r in RESOURCES
                        if known[r] or unknown}
        counts[seat] = int(p['total'])
    event(seats.engine(snap['seat']), {'type': 'jsettlers_hands'}, counts=counts, resource_deltas=deltas)
    return tuple(events)


def reconstruct(snap: dict) -> tuple[Game, Observation, dict]:
    """Return (engine game, observation for our seat, context with the coordinate maps)."""
    decision = snap['decision']
    if decision not in DECISIONS:
        raise SnapshotError(f'unknown decision {decision!r}')
    check_hidden_boundary(snap)
    board, bmap = engine_board(snap['board'])
    seats = Seats(int(snap['first_player']))
    me = seats.engine(int(snap['seat']))
    turn = engine_turn(snap)
    n2v, e2e = bmap.node_to_vertex, bmap.edge_to_engine

    players = [Player(str(s)) for s in range(4)]
    for p in snap['players']:
        seat = seats.engine(p['pn'])
        player = players[seat]
        player.settlements = {n2v[n] for n in p['settlements']}
        player.cities = {n2v[n] for n in p['cities']}
        player.roads = {e2e[e] for e in p['roads']}
        player.knights = int(p['knights'])
        if seat == me:
            player.resources = _hand(snap['own']['resources'])
            dev = []
            for card in snap['own']['dev']:
                name = DEV_CARDS[int(card['type'])]
                dev.append((name, turn if card['state'] == 'new' else turn - 1))
            player.development = dev
            if len(dev) != int(p['dev_count']):
                raise SnapshotError('own development cards do not match the count')
        else:
            player.development = [('knight', -1)] * int(p['dev_count'])
    # Opponent placeholder hands: known cards, unknown ones on the most plentiful bank
    # resource. Only totals (and the engine's bank bookkeeping) depend on them.
    for p in snap['players']:
        seat = seats.engine(p['pn'])
        if seat == me:
            continue
        hand = _hand(p['known'])
        for _ in range(int(p['unknown'])):
            held = {r: sum(pl.resources[r] for pl in players) + hand[r] for r in RESOURCES}
            hand[min(RESOURCES, key=lambda r: held[r])] += 1
        players[seat].resources = hand

    lr, la = snap.get('longest_road', -1), snap.get('largest_army', -1)
    lr_holder = None if lr < 0 else seats.engine(lr)
    la_holder = None if la < 0 else seats.engine(la)
    for s, player in enumerate(players):
        player.longest_road_award = s == lr_holder
        player.largest_army = s == la_holder
    robber_tile = bmap.hex_to_tile[snap['board']['robber']]
    game = Game(board, players, random.Random(0), robber_tile=robber_tile, turn=turn,
                deck=['unknown'] * int(snap.get('dev_remaining', 25)), record_events=False)
    game.longest_road_holder, game.largest_army_holder = lr_holder, la_holder
    occupied = game.occupied()
    game.road_lengths = [longest_road(p.roads, {v for v, o in occupied.items() if o != s})
                         for s, p in enumerate(players)]
    game.played_development = bool(next(p for p in snap['players'] if p['pn'] == snap['seat'])['played_dev'])
    game.current = me
    js_current = seats.engine(int(snap['current']))
    return_phase = 'roll' if snap.get('before_roll') else 'actions'

    if decision == 'setup_settlement':
        game.phase = 'setup_settlement'
        game.setup_index = me if snap.get('setup_round', 1) == 1 else 7 - me
    elif decision == 'setup_road':
        game.phase = 'setup_road'
        game.setup_index = me if snap.get('setup_round', 1) == 1 else 7 - me
        mine = players[me]
        pending = [v for v in mine.settlements if not any(v in e for e in mine.roads)]
        if len(pending) != 1:
            raise SnapshotError(f'cannot identify the settlement awaiting its road: {pending}')
        game.pending_setup_vertex = pending[0]
    elif decision == 'roll':
        game.phase = 'roll'
    elif decision == 'actions':
        game.phase = 'actions'
        game.rolled = int(snap.get('dice', 0)) or None
    elif decision == 'free_roads':
        game.phase = 'free_roads'
        game.free_roads = int(snap.get('free_roads', 2))
        game.return_phase = return_phase
    elif decision == 'robber':
        game.phase = 'robber'
        game.robber_actor = me
        game.return_phase = return_phase
    elif decision == 'steal':
        game.phase = 'steal'
        game.robber_actor = me
        game.return_phase = return_phase
    elif decision == 'discard':
        game.phase = 'discard'
        game.discard_remaining = {me: int(snap['discard_count'])}
        game.robber_actor = js_current
    elif decision == 'trade_response':
        offer = snap['offer']
        game.phase = 'trade_response'
        game.trade_offer = {'owner': seats.engine(offer['from']), 'recipient': me,
                            'give': {r.value: n for r, n in _hand(offer['give']).items() if n},
                            'receive': {r.value: n for r, n in _hand(offer['get']).items() if n}}
    obs = Observation.from_game(game, me, include_events=False)
    obs = replace(obs, events=synthetic_events(snap, seats, turn))
    for p in snap['players']:
        seat = seats.engine(p['pn'])
        if obs.players[seat].public_points != int(p['public_vp']):
            raise SnapshotError(f'public points mismatch for pn {p["pn"]}: engine '
                                f'{obs.players[seat].public_points} vs JSettlers {p["public_vp"]}')
    return game, obs, {'board_map': bmap, 'seats': seats, 'seat': me, 'turn': turn, 'events': obs.events}
