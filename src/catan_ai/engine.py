from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IllegalAction(ValueError):
    pass


class Resource(str, Enum):
    BRICK = "brick"
    LUMBER = "lumber"
    WOOL = "wool"
    GRAIN = "grain"
    ORE = "ore"


RESOURCES = tuple(Resource)
DESERT = "desert"
COSTS = {
    "road": {Resource.BRICK: 1, Resource.LUMBER: 1},
    "settlement": {Resource.BRICK: 1, Resource.LUMBER: 1, Resource.WOOL: 1, Resource.GRAIN: 1},
    "city": {Resource.GRAIN: 2, Resource.ORE: 3},
}


@dataclass(frozen=True)
class Tile:
    q: int
    r: int
    resource: Resource | str
    number: int | None
    vertices: tuple[int, ...]


@dataclass(frozen=True)
class Board:
    tiles: tuple[Tile, ...]
    vertices: tuple[tuple[float, float], ...]
    edges: tuple[tuple[int, int], ...]
    neighbors: tuple[tuple[int, ...], ...]
    harbor_vertices: tuple[int, ...] = ()

    @classmethod
    def random(cls, rng: random.Random) -> "Board":
        coords = [(q, r) for q in range(-2, 3) for r in range(-2, 3) if -2 <= -q - r <= 2]
        terrain: list[Resource | str] = (
            [Resource.BRICK] * 3 + [Resource.LUMBER] * 4 + [Resource.WOOL] * 4
            + [Resource.GRAIN] * 4 + [Resource.ORE] * 3 + [DESERT]
        )
        rng.shuffle(terrain)
        tokens = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        rng.shuffle(tokens)

        points: dict[tuple[int, int], int] = {}
        positions: list[tuple[float, float]] = []
        tile_vertices: list[tuple[int, ...]] = []
        for q, r in coords:
            cx = math.sqrt(3) * (q + r / 2)
            cy = 1.5 * r
            ids = []
            for corner in range(6):
                angle = math.radians(60 * corner - 30)
                x, y = cx + math.cos(angle), cy + math.sin(angle)
                key = (round(x * 1000), round(y * 1000))
                if key not in points:
                    points[key] = len(positions)
                    positions.append((round(x, 4), round(y, 4)))
                ids.append(points[key])
            tile_vertices.append(tuple(ids))

        edge_set: set[tuple[int, int]] = set()
        neighbor_sets = [set() for _ in positions]
        for ids in tile_vertices:
            for i, a in enumerate(ids):
                b = ids[(i + 1) % 6]
                edge = tuple(sorted((a, b)))
                edge_set.add(edge)
                neighbor_sets[a].add(b)
                neighbor_sets[b].add(a)
        edges = tuple(sorted(edge_set))

        token_iter = iter(tokens)
        tiles = tuple(
            Tile(q, r, terrain[i], None if terrain[i] == DESERT else next(token_iter), tile_vertices[i])
            for i, (q, r) in enumerate(coords)
        )
        return cls(tiles, tuple(positions), edges, tuple(tuple(sorted(n)) for n in neighbor_sets))


@dataclass
class Player:
    name: str
    resources: dict[Resource, int] = field(default_factory=lambda: {r: 0 for r in RESOURCES})
    roads: set[tuple[int, int]] = field(default_factory=set)
    settlements: set[int] = field(default_factory=set)
    cities: set[int] = field(default_factory=set)

    @property
    def points(self) -> int:
        return len(self.settlements) + 2 * len(self.cities)

    def can_afford(self, kind: str) -> bool:
        return all(self.resources[r] >= amount for r, amount in COSTS[kind].items())

    def pay(self, kind: str) -> None:
        if not self.can_afford(kind):
            raise IllegalAction(f"cannot afford {kind}")
        for resource, amount in COSTS[kind].items():
            self.resources[resource] -= amount


@dataclass
class Game:
    board: Board
    players: list[Player]
    rng: random.Random = field(repr=False)
    current: int = 0
    phase: str = "setup_settlement"
    setup_index: int = 0
    setup_order: tuple[int, ...] = (0, 1, 2, 3, 3, 2, 1, 0)
    pending_setup_vertex: int | None = None
    rolled: int | None = None
    robber_tile: int = 0
    winner: int | None = None
    turn: int = 0
    last_event: str = "Place your first settlement"

    @classmethod
    def new(cls, names: list[str], seed: int) -> "Game":
        if len(names) != 4:
            raise ValueError("exactly four players are required")
        rng = random.Random(seed)
        board = Board.random(rng)
        robber = next(i for i, tile in enumerate(board.tiles) if tile.resource == DESERT)
        return cls(board, [Player(name) for name in names], rng, robber_tile=robber)

    def occupied(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for seat, player in enumerate(self.players):
            for vertex in player.settlements | player.cities:
                out[vertex] = seat
        return out

    def road_owner(self) -> dict[tuple[int, int], int]:
        return {edge: seat for seat, p in enumerate(self.players) for edge in p.roads}

    def legal_actions(self, seat: int) -> list[dict[str, Any]]:
        if self.winner is not None or seat != self.current:
            return []
        if self.phase == "setup_settlement":
            return [{"type": "build_settlement", "vertex": v} for v in self._legal_settlements(seat, setup=True)]
        if self.phase == "setup_road":
            assert self.pending_setup_vertex is not None
            return [
                {"type": "build_road", "edge": list(sorted((self.pending_setup_vertex, n)))}
                for n in self.board.neighbors[self.pending_setup_vertex]
                if tuple(sorted((self.pending_setup_vertex, n))) not in self.road_owner()
            ]
        if self.phase == "roll":
            return [{"type": "roll"}]
        if self.phase == "robber":
            return [{"type": "move_robber", "tile": i} for i in range(len(self.board.tiles)) if i != self.robber_tile]
        if self.phase != "actions":
            return []
        actions: list[dict[str, Any]] = [{"type": "end_turn"}]
        p = self.players[seat]
        if p.can_afford("road"):
            actions.extend({"type": "build_road", "edge": list(e)} for e in self._legal_roads(seat))
        if p.can_afford("settlement"):
            actions.extend({"type": "build_settlement", "vertex": v} for v in self._legal_settlements(seat))
        if p.can_afford("city"):
            actions.extend({"type": "build_city", "vertex": v} for v in sorted(p.settlements))
        for give in RESOURCES:
            if p.resources[give] >= 4:
                actions.extend(
                    {"type": "bank_trade", "give": give.value, "receive": receive.value}
                    for receive in RESOURCES if receive != give
                )
        return actions

    def apply(self, seat: int, action: dict[str, Any]) -> None:
        legal = self.legal_actions(seat)
        normalized = dict(action)
        if "edge" in normalized:
            normalized["edge"] = list(sorted(int(x) for x in normalized["edge"]))
        if normalized not in legal:
            raise IllegalAction(f"illegal action {normalized}; phase={self.phase}, seat={seat}")
        kind = normalized["type"]
        if kind == "build_settlement":
            self._build_settlement(seat, int(normalized["vertex"]))
        elif kind == "build_road":
            self._build_road(seat, tuple(normalized["edge"]))
        elif kind == "build_city":
            self._build_city(seat, int(normalized["vertex"]))
        elif kind == "roll":
            self._roll()
        elif kind == "move_robber":
            self._move_robber(seat, int(normalized["tile"]))
        elif kind == "bank_trade":
            give, receive = Resource(normalized["give"]), Resource(normalized["receive"])
            self.players[seat].resources[give] -= 4
            self.players[seat].resources[receive] += 1
            self.last_event = f"{self.players[seat].name} traded with the bank"
        elif kind == "end_turn":
            self.current = (self.current + 1) % 4
            self.rolled = None
            self.phase = "roll"
            self.turn += 1
            self.last_event = f"{self.players[self.current].name}'s turn"
        self._check_winner(seat)

    def _legal_settlements(self, seat: int, setup: bool = False) -> list[int]:
        occupied = self.occupied()
        available = [v for v in range(len(self.board.vertices)) if v not in occupied and not any(n in occupied for n in self.board.neighbors[v])]
        if setup:
            return available
        connected = {v for edge in self.players[seat].roads for v in edge}
        return [v for v in available if v in connected]

    def _legal_roads(self, seat: int) -> list[tuple[int, int]]:
        owned = self.road_owner()
        connected = set(self.players[seat].settlements) | set(self.players[seat].cities)
        connected |= {v for edge in self.players[seat].roads for v in edge}
        occupied = self.occupied()
        return [
            edge for edge in self.board.edges
            if edge not in owned and any(v in connected and occupied.get(v, seat) == seat for v in edge)
        ]

    def _build_settlement(self, seat: int, vertex: int) -> None:
        p = self.players[seat]
        if self.phase.startswith("setup"):
            p.settlements.add(vertex)
            self.pending_setup_vertex = vertex
            if self.setup_index >= 4:
                self._grant_adjacent(vertex, seat)
            self.phase = "setup_road"
            self.last_event = f"{p.name} placed a settlement"
        else:
            p.pay("settlement")
            p.settlements.add(vertex)
            self.last_event = f"{p.name} built a settlement"

    def _build_road(self, seat: int, edge: tuple[int, int]) -> None:
        p = self.players[seat]
        if self.phase == "setup_road":
            p.roads.add(edge)
            self.setup_index += 1
            self.pending_setup_vertex = None
            if self.setup_index == len(self.setup_order):
                self.current = 0
                self.phase = "roll"
                self.last_event = "Setup complete"
            else:
                self.current = self.setup_order[self.setup_index]
                self.phase = "setup_settlement"
                self.last_event = f"{self.players[self.current].name} places a settlement"
        else:
            p.pay("road")
            p.roads.add(edge)
            self.last_event = f"{p.name} built a road"

    def _build_city(self, seat: int, vertex: int) -> None:
        p = self.players[seat]
        p.pay("city")
        p.settlements.remove(vertex)
        p.cities.add(vertex)
        self.last_event = f"{p.name} upgraded a city"

    def _grant_adjacent(self, vertex: int, seat: int) -> None:
        for tile in self.board.tiles:
            if vertex in tile.vertices and isinstance(tile.resource, Resource):
                self.players[seat].resources[tile.resource] += 1

    def _roll(self) -> None:
        self.rolled = self.rng.randint(1, 6) + self.rng.randint(1, 6)
        if self.rolled == 7:
            for player in self.players:
                total = sum(player.resources.values())
                if total > 7:
                    for _ in range(total // 2):
                        choices = [r for r, amount in player.resources.items() if amount]
                        player.resources[self.rng.choice(choices)] -= 1
            self.phase = "robber"
            self.last_event = "A 7 was rolled; move the robber"
            return
        occupied = self.occupied()
        for i, tile in enumerate(self.board.tiles):
            if i == self.robber_tile or tile.number != self.rolled or not isinstance(tile.resource, Resource):
                continue
            for vertex in tile.vertices:
                if vertex in occupied:
                    owner = occupied[vertex]
                    amount = 2 if vertex in self.players[owner].cities else 1
                    self.players[owner].resources[tile.resource] += amount
        self.phase = "actions"
        self.last_event = f"{self.players[self.current].name} rolled {self.rolled}"

    def _move_robber(self, seat: int, tile_index: int) -> None:
        self.robber_tile = tile_index
        adjacent = {self.occupied()[v] for v in self.board.tiles[tile_index].vertices if v in self.occupied() and self.occupied()[v] != seat}
        victims = [s for s in adjacent if sum(self.players[s].resources.values())]
        if victims:
            victim = self.rng.choice(victims)
            options = [r for r, amount in self.players[victim].resources.items() if amount]
            stolen = self.rng.choice(options)
            self.players[victim].resources[stolen] -= 1
            self.players[seat].resources[stolen] += 1
        self.phase = "actions"
        self.last_event = f"{self.players[seat].name} moved the robber"

    def _check_winner(self, seat: int) -> None:
        if self.players[seat].points >= 10:
            self.winner = seat
            self.phase = "finished"
            self.last_event = f"{self.players[seat].name} wins"

    def public_state(self, viewer: int | None = None) -> dict[str, Any]:
        legal = self.legal_actions(viewer) if viewer is not None else []
        return {
            "phase": self.phase,
            "current": self.current,
            "turn": self.turn,
            "rolled": self.rolled,
            "robber_tile": self.robber_tile,
            "winner": self.winner,
            "last_event": self.last_event,
            "board": {
                "vertices": self.board.vertices,
                "edges": self.board.edges,
                "tiles": [
                    {"q": t.q, "r": t.r, "resource": t.resource, "number": t.number, "vertices": t.vertices}
                    for t in self.board.tiles
                ],
            },
            "players": [
                {
                    "seat": i,
                    "name": p.name,
                    "points": p.points,
                    "resource_count": sum(p.resources.values()),
                    "resources": ({r.value: n for r, n in p.resources.items()} if i == viewer else None),
                    "roads": sorted(p.roads),
                    "settlements": sorted(p.settlements),
                    "cities": sorted(p.cities),
                }
                for i, p in enumerate(self.players)
            ],
            "legal_actions": legal,
        }
