from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .bots import HeuristicBot
from .engine import Game, IllegalAction


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
    bots: dict[int, HeuristicBot]


sessions: dict[str, Session] = {}


class NewGame(BaseModel):
    name: str = "You"
    seed: int | None = None


class ActionBody(BaseModel):
    action: dict


def advance_bots(session: Session, decision_cap: int = 200) -> None:
    decisions = 0
    while session.game.winner is None and session.game.current != session.human_seat:
        seat = session.game.current
        session.game.apply(seat, session.bots[seat].choose(session.game, seat))
        decisions += 1
        if decisions >= decision_cap:
            raise RuntimeError("bot decision cap reached")


@app.get("/api/healthz")
def health() -> dict:
    return {"ok": True, "games": len(sessions)}


@app.post("/api/games")
def create_game(body: NewGame) -> dict:
    seed = body.seed if body.seed is not None else secrets.randbits(31)
    game_id = secrets.token_urlsafe(8)
    game = Game.new([body.name.strip() or "You", "Ada", "Mina", "Lin"], seed)
    session = Session(
        game=game,
        human_seat=0,
        bots={seat: HeuristicBot(__import__("random").Random(seed * 31 + seat)) for seat in (1, 2, 3)},
    )
    sessions[game_id] = session
    return {"game_id": game_id, "seat": 0, "seed": seed, "state": game.public_state(0)}


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    session = sessions.get(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    return session.game.public_state(session.human_seat)


@app.post("/api/games/{game_id}/actions")
def take_action(game_id: str, body: ActionBody) -> dict:
    session = sessions.get(game_id)
    if session is None:
        raise HTTPException(404, "game not found")
    try:
        session.game.apply(session.human_seat, body.action)
        advance_bots(session)
    except IllegalAction as exc:
        raise HTTPException(400, str(exc)) from exc
    return session.game.public_state(session.human_seat)


def _mount_web() -> None:
    configured = os.environ.get("CATAN_WEB_DIST")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2] / "web" / "dist"
    if root.is_dir():
        app.mount("/", StaticFiles(directory=root, html=True), name="web")


_mount_web()


def main() -> None:
    import uvicorn

    uvicorn.run("catan_ai.api:app", host="0.0.0.0", port=8000, reload=False)
