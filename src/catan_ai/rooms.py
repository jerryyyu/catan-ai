"""Private three-human rooms with one fixed bot and per-seat capabilities."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import secrets
from threading import RLock
import time
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from . import pilot_store
from .bots import make_policy
from .engine import Game, IllegalAction
from .experiments import source_files
from .observation import Observation


router = APIRouter(prefix='/api/rooms')
lock = RLock()
rooms = {}


@dataclass
class Room:
    game: Game
    bot: object
    record: dict


class CreateRoom(BaseModel):
    bot: Literal['tactical', 'heuristic', 'planner_parameter'] = 'tactical'
    bot_seat: int = Field(default=0, ge=0, le=3, strict=True)


class JoinRoom(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    completed_matches: int = Field(ge=0, strict=True)
    platform: str = Field(default='', max_length=80)
    consent: Literal[True]


class RoomAction(BaseModel):
    revision: int = Field(ge=0, strict=True)
    action: dict


def save(room_id, room):
    room.record.update(winner=room.game.winner, turn=room.game.turn)
    pilot_store.save('room_' + room_id, room.record)


def replay_bot(record):
    """Refuse a changed challenger even if it happens to choose the same actions."""
    bot = make_policy(record['bot'], record['seed'] * 31 + record['bot_seat'])
    if record['bot'] == 'planner_parameter':
        if not record.get('model_sha256') or record['model_sha256'] != bot.model_hash:
            raise ValueError('Room checkpoint differs; record is preserved')
    return bot


def load(room_id):
    if room_id in rooms:
        return rooms[room_id]
    try:
        record = pilot_store.load('room_' + room_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, 'Room not found') from exc
    if record['source'] != source_files():
        raise HTTPException(409, 'Room uses a different program version; its record is preserved')
    game = Game.new(record['names'], record['seed'])
    seat = record['bot_seat']
    try:
        bot = replay_bot(record)
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(409, 'Room checkpoint unavailable or changed; record is preserved') from exc
    for row in record['trace']:
        if row['seat'] == seat:
            chosen = bot.choose(Observation.for_policy(game, seat, bot), seat)
            if chosen != row['action']:
                raise HTTPException(409, 'Bot replay differs; record is preserved')
        game.apply(row['seat'], row['action'])
    room = Room(game, bot, record)
    rooms[room_id] = room
    return room


def authorize(room, authorization):
    token = authorization.removeprefix('Bearer ') if authorization else ''
    digest = hashlib.sha256(token.encode()).hexdigest()
    for seat, expected in room.record['seat_hashes'].items():
        if secrets.compare_digest(digest, expected):
            return int(seat)
    raise HTTPException(403, 'A valid seat invitation is required')


def ready(room):
    return len(room.record['participants']) == 3


def advance(room):
    count = 0
    while ready(room) and room.game.winner is None and room.game.current == room.record['bot_seat']:
        started = time.perf_counter()
        seat = room.game.current
        action = room.bot.choose(Observation.for_policy(room.game, seat, room.bot), seat)
        room.game.apply(seat, action)
        room.record['trace'].append({'seat': seat, 'actor': 'bot', 'action': deepcopy(action),
                                     'seconds': time.perf_counter() - started})
        count += 1
        if count >= 200:
            raise RuntimeError('Bot decision cap reached')


def response(room, seat):
    state = room.game.public_state(seat)
    if not ready(room):
        state['legal_actions'] = []
    return {'seat': seat, 'revision': len(room.record['trace']), 'ready': ready(room),
            'joined_seats': sorted(int(s) for s in room.record['participants']),
            'bot_seat': room.record['bot_seat'], 'state': state}


@router.post('')
def create(body: CreateRoom):
    room_id, seed = secrets.token_urlsafe(12), secrets.randbits(31)
    tokens = {str(s): secrets.token_urlsafe(32) for s in range(4) if s != body.bot_seat}
    names = [f'Human {s + 1}' if s != body.bot_seat else body.bot.title() for s in range(4)]
    game = Game.new(names, seed)
    record = {'schema': 1, 'purpose': 'private three-human pilot; not confirmatory transfer',
              'seed': seed, 'names': names, 'bot': body.bot, 'bot_seat': body.bot_seat,
              'source': source_files(), 'created_at': datetime.now(timezone.utc).isoformat(),
              'participants': {}, 'trace': [],
              'seat_hashes': {s: hashlib.sha256(t.encode()).hexdigest() for s, t in tokens.items()}}
    try:
        bot = make_policy(body.bot, seed * 31 + body.bot_seat)
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(503, 'Requested bot checkpoint is not configured for this server version') from exc
    if body.bot == 'planner_parameter':
        record['model_sha256'] = bot.model_hash
    room = Room(game, bot, record)
    with lock:
        save(room_id, room)
        rooms[room_id] = room
    return {'room_id': room_id, 'invitations': tokens, 'bot_seat': body.bot_seat}


@router.post('/{room_id}/join')
def join(room_id: str, body: JoinRoom, authorization: str | None = Header(default=None)):
    with lock:
        room = load(room_id)
        seat = authorize(room, authorization)
        participant = body.model_dump()
        previous = room.record['participants'].get(str(seat))
        if previous is not None and previous != participant:
            raise HTTPException(409, 'This seat has already registered')
        room.record['participants'][str(seat)] = participant
        room.record['names'][seat] = body.name
        room.game.players[seat].name = body.name
        try:
            advance(room)
        finally:
            save(room_id, room)
        return response(room, seat)


@router.get('/{room_id}')
def state(room_id: str, authorization: str | None = Header(default=None)):
    with lock:
        room = load(room_id)
        return response(room, authorize(room, authorization))


@router.post('/{room_id}/actions')
def act(room_id: str, body: RoomAction, authorization: str | None = Header(default=None)):
    with lock:
        room = load(room_id)
        seat = authorize(room, authorization)
        if not ready(room) or body.revision != len(room.record['trace']):
            raise HTTPException(409, 'Wait for all players or refresh the current position')
        try:
            started = time.perf_counter()
            room.game.apply(seat, body.action)
            room.record['trace'].append({'seat': seat, 'actor': 'human', 'action': deepcopy(body.action),
                                         'seconds': time.perf_counter() - started})
            advance(room)
        except IllegalAction as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            save(room_id, room)
        return response(room, seat)


@router.get('/{room_id}/replay')
def replay(room_id: str, step: int | None = Query(default=None, ge=0),
           authorization: str | None = Header(default=None)):
    with lock:
        room = load(room_id)
        seat = authorize(room, authorization)
        trace = room.record['trace']
        selected = len(trace) if step is None else step
        if selected > len(trace):
            raise HTTPException(422, 'Replay step exceeds the recorded match')
        game = Game.new(room.record['names'], room.record['seed'])
        for row in trace[:selected]:
            game.apply(row['seat'], row['action'])
        return {'seat': seat, 'step': selected, 'total_steps': len(trace),
                'purpose': room.record['purpose'], 'state': game.public_state(seat),
                'events': game.observed_events(seat),
                'timings': [{k: row[k] for k in ('seat', 'actor', 'seconds')} for row in trace[:selected]]}
