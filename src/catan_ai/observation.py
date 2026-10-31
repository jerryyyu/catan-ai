"""Detached policy inputs containing only the acting player's information."""

from copy import deepcopy
from dataclasses import dataclass

from .engine import Board, Game, Resource


@dataclass(frozen=True)
class PlayerView:
    resources: dict[Resource, int] | None
    resource_count: int
    public_points: int
    settlements: frozenset[int]
    cities: frozenset[int]
    roads: frozenset[tuple[int, int]]
    development: tuple[str, ...] | None
    development_count: int
    development_turns: tuple[int, ...] | None
    knights: int


@dataclass(frozen=True)
class Observation:
    seat: int
    phase: str
    turn: int
    board: Board
    players: tuple[PlayerView, ...]
    trade_offer: dict | None
    actions: tuple[dict, ...]
    rates: dict[Resource, int]
    robber_tile: int
    rolled: int | None
    setup_index: int
    pending_setup_vertex: int | None
    discard_remaining: dict[int, int]
    robber_actor: int | None
    longest_road_holder: int | None
    largest_army_holder: int | None
    played_development: bool
    return_phase: str
    free_roads: int
    events: tuple[dict, ...]

    @classmethod
    def from_game(cls, game: Game, seat: int, *, include_events: bool = True) -> 'Observation':
        players = tuple(PlayerView(
            dict(p.resources) if i == seat else None,
            sum(p.resources.values()), p.public_points,
            frozenset(p.settlements), frozenset(p.cities), frozenset(p.roads),
            tuple(card for card, _ in p.development) if i == seat else None,
            len(p.development), tuple(turn for _, turn in p.development) if i == seat else None,
            p.knights,
        ) for i, p in enumerate(game.players))
        return cls(seat, game.phase, game.turn, game.board, players,
                   deepcopy(game.trade_offer), tuple(game.legal_actions(seat)),
                   game.trade_rates(seat),
                   game.robber_tile, game.rolled, game.setup_index, game.pending_setup_vertex,
                   dict(game.discard_remaining), game.robber_actor, game.longest_road_holder,
                   game.largest_army_holder, game.played_development, game.return_phase,
                   game.free_roads, tuple(game.observed_events(seat)) if include_events else ())

    @classmethod
    def for_policy(cls, game: Game, seat: int, policy) -> 'Observation':
        # New policies receive full history unless they explicitly opt out.
        return cls.from_game(game, seat, include_events=getattr(policy, 'needs_history', True))

    def legal_actions(self, seat: int) -> list[dict]:
        return [_copy_action(action) for action in self.actions] if seat == self.seat else []

    def occupied(self) -> dict[int, int]:
        return {v: s for s, p in enumerate(self.players) for v in p.settlements | p.cities}

    def trade_rate(self, seat: int, resource: Resource) -> int:
        if seat != self.seat:
            raise ValueError('trade rates are provided for the observing player')
        return self.rates[resource]


def _copy_action(action):
    # Legal-action payloads contain scalar fields and flat scalar lists. Retain
    # a general copy fallback so future nested payloads stay detached too.
    return {key: (list(value) if isinstance(value, list)
                  and all(isinstance(item, (str, int, float, bool, type(None))) for item in value)
                  else deepcopy(value) if isinstance(value, (dict, list, set, tuple)) else value)
            for key, value in action.items()}
