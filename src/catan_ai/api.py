from __future__ import annotations

import logging
import os
import re
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bots import Policy, make_policy
from .engine import Game, IllegalAction
from .observation import Observation
from . import pilot_store

from .experiments import source_files
from .local_checkpoint import CYTHON_PARAMETER_CHECKPOINT, select_parameter_checkpoint  # noqa: F401

select_parameter_checkpoint()


app = FastAPI(title="Catan AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Session:
    game: Game
    human_seat: int
    bots: dict[int, Policy]
    game_id: str | None = None
    metadata: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


sessions: dict[str, Session] = {}
session_lock = RLock()


def persist(session):
    if session.game_id is not None:
        pilot_store.save(session.game_id, {**session.metadata, 'trace': session.trace,
                                          'winner': session.game.winner, 'turn': session.game.turn})


def load_session(game_id):
    if game_id in sessions:
        return sessions[game_id]
    try:
        record = pilot_store.load(game_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, 'game not found') from exc
    if record['source'] != source_files():
        raise HTTPException(409, 'Saved match uses a different program version; its record is preserved.')
    if record['opponent'] not in ('heuristic', 'tactical') + CHECKPOINT_POLICIES:
        raise HTTPException(409, 'Restart recovery supports Tactical, Heuristic and trained challenger pilots.')
    game = Game.new(record['names'], record['seed'])
    human_seat = record.get('human_seat', 0)
    try:
        bots = {seat: make_policy(record['opponent'], record['seed'] * 31 + seat)
                for seat in range(4) if seat != human_seat}
        if record['opponent'] in CHECKPOINT_POLICIES:
            if not record.get('model_sha256') or any(
                    bot.model_hash != record['model_sha256'] for bot in bots.values()):
                raise ValueError('Saved checkpoint differs')
        if record['opponent'] in VALUE_WEIGHT_POLICIES:
            if not record.get('value_sha256') or any(
                    value_weights_hash(bot) != record['value_sha256'] for bot in bots.values()):
                raise ValueError('Saved value weights differ')
    except (KeyError, OSError, ValueError, ImportError) as exc:
        raise HTTPException(409, 'Saved checkpoint unavailable or changed; record is preserved.') from exc
    for row in record['trace']:
        seat = row['seat']
        if seat != human_seat:
            chosen = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
            if chosen != row['action']:
                raise HTTPException(409, 'Saved bot decisions do not replay; record is preserved.')
        game.apply(seat, row['action'])
    session = Session(game, human_seat, bots, game_id,
                      {k: v for k, v in record.items() if k not in ('trace', 'winner', 'turn')}, record['trace'])
    sessions[game_id] = session
    return session


# Policies that load the source-pinned fitted checkpoint; their games record its SHA256.
CHECKPOINT_POLICIES = ('planner_parameter', 'planner_devfix', 'planner_unified', 'planner_unified_offer',
                       'planner_offer_guard', 'planner_offer_accept', 'planner_pv2_tiles',
                       'planner_pv2_tiles_va_t9')
# Policies that also load local value-net weights (never committed; pinned by the
# SHA256 in their .meta.json); their games record that SHA256 too.
VALUE_WEIGHT_POLICIES = ('planner_pv2_tiles', 'planner_pv2_tiles_va_t9')


def value_weights_hash(bot) -> str:
    return bot.net.meta['npz_sha256']


class NewGame(BaseModel):
    name: str = "You"
    seed: int | None = None
    human_seat: int = Field(default=0, ge=0, le=3, strict=True)
    # None means the served default (see served_default).
    opponent: Literal['heuristic', 'mc', 'tactical', 'mc_mean', 'learned', 'planner_parameter',
                      'planner_devfix', 'planner_unified', 'planner_unified_offer',
                      'planner_offer_guard', 'planner_offer_accept', 'planner_pv2_tiles',
                      'planner_pv2_tiles_va_t9'] | None = None


# Since 2026-11-01 (owner-approved): the shared-score planner plus trade offers.
# planner_unified keeps the fitted build search and scores robber, steal, discard,
# development-card and trade-answer moves with one position score
# (runs/planner-unified-heldout-v1.json, vs planner_parameter); this arm also
# proposes at most one player trade per turn and passed its held-out test against
# planner_unified (runs/planner-offer-heldout-v1.json). Since 2026-11-02 the offers
# skip seats on 8+ public points and are priced net of the recipient's gain
# (planner_offer_guard; observed lift above a -.03 point floor vs planner_unified_offer,
# runs/offer-guard-dev-v1.json; no held-out of its own). Since 2026-11-03 each offer is
# weighted by the recipient's public acceptance rate (planner_offer_accept; held-out
# mixed-table PASS vs planner_offer_guard, runs/offer-accept-heldout-v1.json).
# Since 2026-11-08 (owner promotion, Jerry): the same planner with its own-turn build
# decisions chosen by PV search scored by the tile-token value transformer
# (planner_pv2_tiles; precise 512-seed lift vs planner_offer_accept on all three
# tables, runs/pv2-tiles-precise-v1; the predeclared 80-seed screen FAILED its
# all-tables +.05 rule and there is no held-out run). Its weights are a local file
# pinned by SHA256; without them new games fall back to planner_offer_accept.
# Since 2026-11-11 (owner promotion, Jerry): planner_pv2_tiles_buyfix scored by the
# retrained value-arms T9 transformer (planner_pv2_tiles_va_t9; held-out PASS on both
# tables vs planner_pv2_tiles, self table +.035 LCB +.024, JSettlers +.026 LCB +.012).
# Its weights are their own local file pinned by SHA256 and by the gate-scored .pt
# SHA256; without them new games fall back to planner_pv2_tiles, then as above.
# History: docs/POLICIES.md.
SERVED_POLICY = 'planner_pv2_tiles_va_t9'
VALUE_FALLBACK_POLICY = 'planner_pv2_tiles'
CHECKPOINT_FALLBACK_POLICY = 'planner_offer_accept'
FALLBACK_POLICY = 'tactical'
CHECKPOINT_UNBOUND = 'Trained planner checkpoint is not bound to this server version'
T9_WEIGHTS_UNAVAILABLE = 'T9 value weights missing or unverified'
VALUE_WEIGHTS_UNAVAILABLE = 'Tiles value weights missing or unverified'

# (stem, npz stat, meta stat) -> (available, detail). The weights are verified (SHA256,
# tokenizer, parity probe, and for T9 the gate-scored .pt SHA256) once per file
# version per stem, not once per request.
_value_weights_cache: dict[tuple, tuple[bool, str | None]] = {}


def _tiles_weights():
    from . import tiles_value
    return tiles_value.TILES_V1, tiles_value.TilesValueNet


def _t9_weights():
    from . import tiles_value, value_arms_serving
    stem = value_arms_serving.CHECKPOINTS / 'value_tiles_va_t9'

    def verify(path):
        # The same pin ValueArmPlanner enforces when it builds the served bot.
        net = tiles_value.TilesValueNet(path)
        if net.meta.get('pt_sha256') != value_arms_serving.ARMS['t9'] or net.meta.get('value_arm') != 't9':
            raise ValueError('T9 value weights were not exported from the gate-scored checkpoint')
    return stem, verify


# Policy -> () -> (weights stem, verifier that raises on a bad export).
_VALUE_WEIGHTS = {'planner_pv2_tiles': _tiles_weights, 'planner_pv2_tiles_va_t9': _t9_weights}


def _stat_key(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino


def value_weights_status(policy: str = VALUE_FALLBACK_POLICY) -> tuple[bool, str | None]:
    """Return (available, detail) for one policy's local value weights; never raises."""
    try:
        stem, verify = _VALUE_WEIGHTS[policy]()
        stem = str(stem)
        key = (stem, _stat_key(Path(f'{stem}.npz')), _stat_key(Path(f'{stem}.meta.json')))
        if key not in _value_weights_cache:
            import numpy  # noqa: F401  (the scorer is NumPy; report its absence by name)
            verify(stem)
            for old in [k for k in _value_weights_cache if k[0] == stem]:
                del _value_weights_cache[old]
            _value_weights_cache[key] = (True, None)
    except Exception as exc:  # any load failure is a visible fallback, not a crash
        return False, type(exc).__name__ + ': ' + str(exc)
    return _value_weights_cache[key]


_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:)?[/\\][^\s'\"]*")
_logged_details: set[str] = set()


def public_detail(detail: str | None) -> str | None:
    """A fallback detail safe for /api/bots: absolute paths cut to their basename.

    The full detail goes to the server log, once per distinct message.
    """
    if detail is None:
        return None
    if detail not in _logged_details:
        _logged_details.add(detail)
        logging.getLogger(__name__).warning('served bot fallback: %s', detail)
    return _ABSOLUTE_PATH.sub(lambda m: m.group().replace('\\', '/').rstrip('/').rsplit('/', 1)[-1], detail)


def serving_status() -> tuple[str, str | None, str | None]:
    """The policy a new game gets when none is named, and why if it fell back.

    Returns (policy, reason, detail). ``planner_pv2_tiles_va_t9`` is served when the
    source-pinned parameter checkpoint loads on this server (pure-Python engine,
    matching source) and its local T9 value weights load and verify. Without them
    new games use ``planner_pv2_tiles`` (its own tiles weights); without those (or
    NumPy) ``planner_offer_accept``; without the checkpoint Tactical. Each fallback
    has its own reason so it is visible.
    """
    from .parameter_planner import served_status
    available, detail = served_status()
    if not available:
        return FALLBACK_POLICY, CHECKPOINT_UNBOUND, detail
    available, t9_detail = value_weights_status(SERVED_POLICY)
    if available:
        return SERVED_POLICY, None, None
    available, detail = value_weights_status(VALUE_FALLBACK_POLICY)
    if available:
        return VALUE_FALLBACK_POLICY, T9_WEIGHTS_UNAVAILABLE, t9_detail
    return CHECKPOINT_FALLBACK_POLICY, VALUE_WEIGHTS_UNAVAILABLE, detail


def served_default() -> tuple[str, str | None]:
    """(policy, reason) for new games that name no opponent; see serving_status."""
    policy, reason, _ = serving_status()
    return policy, reason


class ActionBody(BaseModel):
    action: dict


def advance_bots(session: Session, decision_cap: int = 200) -> None:
    decisions = 0
    while session.game.winner is None and session.game.current != session.human_seat:
        seat = session.game.current
        started = time.perf_counter()
        action = session.bots[seat].choose(Observation.for_policy(session.game, seat, session.bots[seat]), seat)
        session.game.apply(seat, action)
        session.trace.append({'seat': seat, 'action': deepcopy(action),
                              'seconds': time.perf_counter() - started, 'actor': 'bot'})
        decisions += 1
        if decisions >= decision_cap:
            raise RuntimeError("bot decision cap reached")


@app.get("/api/healthz")
def health() -> dict:
    return {"ok": True, "games": len(sessions)}


@app.get("/api/bots")
def bots_info() -> dict:
    policy, reason, detail = serving_status()
    checkpoint_available = policy != FALLBACK_POLICY
    return {"default": policy, "served": SERVED_POLICY, "served_available": reason is None,
            "fallback_reason": reason, "fallback_detail": public_detail(detail),
            "checkpoint_available": checkpoint_available,
            # Whether the previous default (planner_pv2_tiles) can still be chosen by name.
            "tiles_available": checkpoint_available and value_weights_status(VALUE_FALLBACK_POLICY)[0]}


@app.post("/api/games")
def create_game(body: NewGame) -> dict:
    if body.opponent is None:
        body = body.model_copy(update={'opponent': served_default()[0]})
    seed = body.seed if body.seed is not None else secrets.randbits(31)
    game_id = secrets.token_urlsafe(8)
    names = ["Ada", "Mina", "Lin"]
    names.insert(body.human_seat, body.name.strip() or "You")
    game = Game.new(names, seed)
    try:
        bots = {seat: make_policy(body.opponent, seed * 31 + seat)
                for seat in range(4) if seat != body.human_seat}
        hashes = {bot.model_hash for bot in bots.values()} if body.opponent in CHECKPOINT_POLICIES else set()
        if hashes and len(hashes) != 1:
            raise ValueError('Checkpoint changed during game creation')
        value_hashes = ({value_weights_hash(bot) for bot in bots.values()}
                        if body.opponent in VALUE_WEIGHT_POLICIES else set())
        if value_hashes and len(value_hashes) != 1:
            raise ValueError('Value weights changed during game creation')
    except (KeyError, OSError, ValueError, ImportError) as exc:
        raise HTTPException(503, 'Requested bot checkpoint is not configured for this server version') from exc
    session = Session(
        game=game,
        human_seat=body.human_seat,
        bots=bots,
        game_id=game_id,
        metadata={'schema': 2, 'seed': seed, 'human_seat': body.human_seat,
                  'names': [p.name for p in game.players],
                  'opponent': body.opponent, 'source': source_files(),
                  'created_at': datetime.now(timezone.utc).isoformat(),
                  'purpose': 'local pilot; not confirmatory human transfer'},
    )
    if hashes:
        session.metadata['model_sha256'] = next(iter(hashes))
    if value_hashes:
        session.metadata['value_sha256'] = next(iter(value_hashes))
    with session_lock:
        persist(session)
        sessions[game_id] = session
        try:
            advance_bots(session)
        finally:
            persist(session)
    return {"game_id": game_id, "seat": body.human_seat, "state": game.public_state(body.human_seat),
            "opponent": body.opponent,
            "model_loaded": (all(getattr(bot, 'model', None) is not None for bot in session.bots.values())
                             if body.opponent == 'learned' else None)}


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    with session_lock:
        session = load_session(game_id)
        advance_bots(session)
        persist(session)
        return session.game.public_state(session.human_seat)


@app.get('/api/games/{game_id}/replay')
def export_replay(game_id: str, step: int | None = Query(default=None, ge=0)) -> dict:
    with session_lock:
        session = load_session(game_id)
        total_steps = len(session.trace)
        selected = total_steps if step is None else step
        if selected > total_steps:
            raise HTTPException(422, 'Replay step exceeds the recorded match.')
        game = session.game
        if selected != total_steps:
            game = Game.new(session.metadata['names'], session.metadata['seed'])
            for row in session.trace[:selected]:
                game.apply(row['seat'], row['action'])
        return {'schema': 1, 'purpose': session.metadata['purpose'],
                'opponent': session.metadata['opponent'], 'created_at': session.metadata['created_at'],
                'human_seat': session.human_seat,
                'step': selected, 'total_steps': total_steps,
                'completed': game.winner is not None, 'state': game.public_state(session.human_seat),
                'events': game.observed_events(session.human_seat),
                'timings': [{'seat': r['seat'], 'actor': r['actor'], 'seconds': r['seconds']}
                            for r in session.trace[:selected]]}


@app.post("/api/games/{game_id}/actions")
def take_action(game_id: str, body: ActionBody) -> dict:
    with session_lock:
        session = load_session(game_id)
        try:
            started = time.perf_counter()
            session.game.apply(session.human_seat, body.action)
            session.trace.append({'seat': session.human_seat, 'action': deepcopy(body.action),
                                  'seconds': time.perf_counter() - started, 'actor': 'human'})
            advance_bots(session)
        except IllegalAction as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            persist(session)
        return session.game.public_state(session.human_seat)


def _mount_web() -> None:
    configured = os.environ.get("CATAN_WEB_DIST")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2] / "web" / "dist"
    if root.is_dir():
        app.mount("/", StaticFiles(directory=root, html=True), name="web")


from .rooms import router as room_router  # noqa: E402  (router needs `app` defined above)

app.include_router(room_router)
_mount_web()


def main() -> None:
    import uvicorn

    uvicorn.run("catan_ai.api:app", host="0.0.0.0", port=8000, reload=False)
