"""Authoritative rules source, executable as Python or compiled with Cython."""
from __future__ import annotations

import math
import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

if __name__.endswith('._engine_cy'):
    from ._scoring_cy import award_holder, longest_road
else:
    from .scoring import award_holder, longest_road


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
    "development": {Resource.GRAIN: 1, Resource.ORE: 1, Resource.WOOL: 1},
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
    ports: tuple[tuple[tuple[int, int], Resource | None], ...] = ()

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

        adjacent_pairs = [(a, b) for a in range(19) for b in range(a + 1, 19)
                          if len(set(tile_vertices[a]) & set(tile_vertices[b])) == 2]
        for _ in range(10000):
            token_iter = iter(tokens)
            tiles = tuple(
                Tile(q, r, terrain[i], None if terrain[i] == DESERT else next(token_iter), tile_vertices[i])
                for i, (q, r) in enumerate(coords)
            )
            if not any(tiles[a].number in (6, 8) and tiles[b].number in (6, 8) for a, b in adjacent_pairs):
                break
            rng.shuffle(tokens)
        else:
            raise RuntimeError('could not generate a board with separated red tokens')
        coast = [edge for edge in edges if sum(set(edge) <= set(ids) for ids in tile_vertices) == 1]
        coast.sort(key=lambda e: math.atan2(sum(positions[v][1] for v in e), sum(positions[v][0] for v in e)))
        port_types = [None] * 4 + list(RESOURCES)
        rng.shuffle(port_types)
        ports = tuple((coast[i * len(coast) // 9], resource) for i, resource in enumerate(port_types))
        return cls(tiles, tuple(positions), edges, tuple(tuple(sorted(n)) for n in neighbor_sets), ports)


@dataclass
class Player:
    name: str
    resources: dict[Resource, int] = field(default_factory=lambda: {r: 0 for r in RESOURCES})
    roads: set[tuple[int, int]] = field(default_factory=set)
    settlements: set[int] = field(default_factory=set)
    cities: set[int] = field(default_factory=set)
    longest_road_award: bool = False
    development: list[tuple[str, int]] = field(default_factory=list)
    knights: int = 0
    largest_army: bool = False

    @property
    def points(self) -> int:
        return self.public_points + sum(card == "victory_point" for card, _ in self.development)

    @property
    def public_points(self) -> int:
        return len(self.settlements) + 2 * len(self.cities) + 2 * self.longest_road_award + 2 * self.largest_army

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
    discard_remaining: dict[int, int] = field(default_factory=dict)
    robber_actor: int | None = None
    longest_road_holder: int | None = None
    road_lengths: list[int] = field(default_factory=lambda: [0] * 4)
    deck: list[str] = field(default_factory=list)
    played_development: bool = False
    return_phase: str = "actions"
    free_roads: int = 0
    largest_army_holder: int | None = None
    trade_offer: dict | None = None
    events: list[dict] = field(default_factory=list)
    record_events: bool = True

    @classmethod
    def new(cls, names: list[str], seed: int) -> "Game":
        if len(names) != 4:
            raise ValueError("exactly four players are required")
        rng = random.Random(seed)
        board = Board.random(rng)
        robber = next(i for i, tile in enumerate(board.tiles) if tile.resource == DESERT)
        deck = ["knight"] * 14 + ["victory_point"] * 5 + ["monopoly", "year_of_plenty", "road_building"] * 2
        rng.shuffle(deck)
        return cls(board, [Player(name) for name in names], rng, robber_tile=robber, deck=deck)

    def occupied(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for seat, player in enumerate(self.players):
            for vertex in player.settlements | player.cities:
                out[vertex] = seat
        return out

    def road_owner(self) -> dict[tuple[int, int], int]:
        return {edge: seat for seat, p in enumerate(self.players) for edge in p.roads}

    @property
    def bank(self) -> dict[Resource, int]:
        """Cards not held by players remain in the 19-card supply."""
        return {r: 19 - sum(p.resources[r] for p in self.players) for r in RESOURCES}

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
            return [{"type": "roll"}] + self._development_actions(seat)
        if self.phase == "trade_response":
            assert self.trade_offer is not None
            actions = [{"type": "reject_trade"}]
            if all(self.players[seat].resources[Resource(r)] >= n for r, n in self.trade_offer["receive"].items()):
                actions.append({"type": "accept_trade"})
            return actions
        if self.phase == "free_roads":
            return [{"type": "build_road", "edge": list(e)} for e in self._legal_roads(seat)]
        if self.phase == "discard":
            return [{"type": "discard", "resource": r.value} for r in RESOURCES if self.players[seat].resources[r]]
        if self.phase == "steal":
            return [{"type": "steal", "victim": victim} for victim in self._robber_victims(seat)]
        if self.phase == "robber":
            return [{"type": "move_robber", "tile": i} for i in range(len(self.board.tiles)) if i != self.robber_tile]
        if self.phase != "actions":
            return []
        actions: list[dict[str, Any]] = [{"type": "end_turn"}] + self._development_actions(seat)
        p = self.players[seat]
        if self.deck and p.can_afford("development"):
            actions.append({"type": "buy_development"})
        if p.can_afford("road"):
            actions.extend({"type": "build_road", "edge": list(e)} for e in self._legal_roads(seat))
        if p.can_afford("settlement"):
            actions.extend({"type": "build_settlement", "vertex": v} for v in self._legal_settlements(seat))
        if p.can_afford("city") and len(p.cities) < 4:
            actions.extend({"type": "build_city", "vertex": v} for v in sorted(p.settlements))
        rates = self.trade_rates(seat)
        bank = None
        for give in RESOURCES:
            if p.resources[give] >= rates[give]:
                if bank is None:
                    bank = self.bank
                actions.extend(
                    {"type": "bank_trade", "give": give.value, "receive": receive.value}
                    for receive in RESOURCES if receive != give and bank[receive] > 0
                )
        return actions

    def _development_actions(self, seat: int) -> list[dict[str, Any]]:
        if self.played_development:
            return []
        cards = {card for card, bought in self.players[seat].development if bought < self.turn}
        actions = []
        for card in sorted(cards):
            if card == "knight":
                actions.append({"type": "play_development", "card": card})
            elif card == "road_building" and self._legal_roads(seat):
                actions.append({"type": "play_development", "card": card})
            elif card == "monopoly":
                actions.extend({"type": "play_development", "card": card, "resource": r.value} for r in RESOURCES)
            elif card == "year_of_plenty":
                bank = self.bank
                for i, first in enumerate(RESOURCES):
                    for second in RESOURCES[i:]:
                        if bank[first] >= 1 + (first == second) and bank[second] >= 1:
                            actions.append({"type": "play_development", "card": card, "resources": [first.value, second.value]})
                if sum(bank.values()) == 1:
                    actions.append({"type": "play_development", "card": card, "resources": [next(r.value for r in RESOURCES if bank[r])]})
        return actions

    def _play_development(self, seat: int, action: dict) -> None:
        player = self.players[seat]
        card = action["card"]
        owned = next(c for c in player.development if c[0] == card and c[1] < self.turn)
        player.development.remove(owned)
        self.played_development = True
        self.return_phase = self.phase
        if card == "knight":
            player.knights += 1
            self.largest_army_holder = award_holder([p.knights for p in self.players], 3, self.largest_army_holder)
            for i, p in enumerate(self.players):
                p.largest_army = i == self.largest_army_holder
            self.phase = "robber"
        elif card == "road_building":
            self.free_roads = 2
            self.phase = "free_roads"
        elif card == "monopoly":
            resource = Resource(action["resource"])
            for i, other in enumerate(self.players):
                if i != seat:
                    player.resources[resource] += other.resources[resource]
                    other.resources[resource] = 0
        elif card == "year_of_plenty":
            for resource in action["resources"]:
                player.resources[Resource(resource)] += 1
        self.last_event = f"{player.name} played {card.replace('_', ' ')}"

    def trade_rate(self, seat: int, resource: Resource) -> int:
        buildings = self.players[seat].settlements | self.players[seat].cities
        rates = [4]
        for vertices, kind in self.board.ports:
            if buildings.intersection(vertices):
                if kind is None:
                    rates.append(3)
                elif kind == resource:
                    rates.append(2)
        return min(rates)

    def trade_rates(self, seat: int) -> dict[Resource, int]:
        buildings = self.players[seat].settlements | self.players[seat].cities
        generic = 4
        specific = set()
        for vertices, kind in self.board.ports:
            if buildings.intersection(vertices):
                if kind is None:
                    generic = 3
                else:
                    specific.add(kind)
        return {r: 2 if r in specific else generic for r in RESOURCES}

    def apply(self, seat: int, action: dict[str, Any]) -> None:
        if not self.record_events:
            self._apply(seat, action)
            return
        before = [dict(p.resources) for p in self.players]
        phase, turn = self.phase, self.turn
        self._apply(seat, action)
        kind = action['type']
        visible_action = deepcopy(action)
        if kind == 'discard':
            visible_action.pop('resource')
        deltas = [{r.value: p.resources[r] - old[r] for r in RESOURCES
                   if p.resources[r] != old[r]} for p, old in zip(self.players, before)]
        public = {'seat': seat, 'turn': turn, 'phase': phase, 'action': visible_action,
                  'counts': [sum(p.resources.values()) for p in self.players],
                  'public_points': [p.public_points for p in self.players],
                  'winner': self.winner,
                  'turn_owner': (self.trade_offer['owner'] if self.trade_offer else
                                 self.robber_actor if self.phase == 'discard' else self.current)}
        private = {}
        if kind in {'discard', 'steal'}:
            informed = {seat}
            if kind == 'steal':
                informed.add(action['victim'])
            private = {s: {'resource_deltas': deltas} for s in informed}
        else:
            public['resource_deltas'] = deltas
        if kind == 'roll':
            public['rolled'] = self.rolled
            if self.rolled != 7:
                claims = [{r.value: 0 for r in RESOURCES} for _ in self.players]
                occupied = self.occupied()
                for i, tile in enumerate(self.board.tiles):
                    if i == self.robber_tile or tile.number != self.rolled or not isinstance(tile.resource, Resource):
                        continue
                    for vertex in tile.vertices:
                        if vertex in occupied:
                            owner = occupied[vertex]
                            claims[owner][tile.resource.value] += 2 if vertex in self.players[owner].cities else 1
                public['production_claims'] = claims
        if kind == 'buy_development':
            private[seat] = {'drawn_card': self.players[seat].development[-1][0]}
        self.events.append({'public': public, 'private': private})

    def observed_events(self, viewer: int | None) -> list[dict]:
        return [deepcopy(event['public'] | event['private'].get(viewer, {}))
                for event in self.events]

    def _apply(self, seat: int, action: dict[str, Any]) -> None:
        if type(seat) is not int or seat not in range(4) or not isinstance(action, dict):
            raise IllegalAction('invalid actor or action')
        if action.get("type") == "offer_trade":
            self._offer_trade(seat, action)
            return
        legal = self.legal_actions(seat)
        normalized = dict(action)
        if "edge" in normalized:
            edge = normalized['edge']
            if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(type(v) is not int for v in edge):
                raise IllegalAction('edge requires two integer vertices')
            normalized["edge"] = sorted(edge)
        try:
            encoded = json.dumps(normalized, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise IllegalAction('action must contain valid JSON values') from exc
        if not any(normalized == a and encoded == json.dumps(a, sort_keys=True) for a in legal):
            raise IllegalAction(f"illegal action {normalized}; phase={self.phase}, seat={seat}")
        kind = normalized["type"]
        if kind in {"accept_trade", "reject_trade"}:
            assert self.trade_offer is not None
            offer = self.trade_offer
            owner = offer["owner"]
            if kind == "accept_trade":
                for resource, amount in offer["give"].items():
                    self.players[owner].resources[Resource(resource)] -= amount
                    self.players[seat].resources[Resource(resource)] += amount
                for resource, amount in offer["receive"].items():
                    self.players[seat].resources[Resource(resource)] -= amount
                    self.players[owner].resources[Resource(resource)] += amount
            self.current = owner
            self.phase = "actions"
            self.trade_offer = None
            self.last_event = f"{self.players[seat].name} {'accepted' if kind == 'accept_trade' else 'rejected'} the trade"
        elif kind == "buy_development":
            self.players[seat].pay("development")
            self.players[seat].development.append((self.deck.pop(), self.turn))
            self.last_event = f"{self.players[seat].name} bought a development card"
        elif kind == "play_development":
            self._play_development(seat, normalized)
        elif kind == "build_settlement":
            self._build_settlement(seat, int(normalized["vertex"]))
        elif kind == "build_road":
            self._build_road(seat, tuple(normalized["edge"]))
        elif kind == "build_city":
            self._build_city(seat, int(normalized["vertex"]))
        elif kind == "roll":
            self._roll()
        elif kind == "move_robber":
            self._move_robber(seat, int(normalized["tile"]))
        elif kind == "discard":
            self.players[seat].resources[Resource(normalized["resource"])] -= 1
            self.discard_remaining[seat] -= 1
            if not self.discard_remaining[seat]:
                del self.discard_remaining[seat]
            if self.discard_remaining:
                self.current = next(iter(self.discard_remaining))
            else:
                assert self.robber_actor is not None
                self.current = self.robber_actor
                self.phase = "robber"
            self.last_event = "Choose discards" if self.discard_remaining else "Move the robber"
        elif kind == "steal":
            victim = self.players[int(normalized["victim"])]
            cards = [r for r, amount in victim.resources.items() for _ in range(amount)]
            resource = self.rng.choice(cards)
            victim.resources[resource] -= 1
            self.players[seat].resources[resource] += 1
            self.phase = self.return_phase
            self.last_event = f"{self.players[seat].name} stole a resource"
        elif kind == "bank_trade":
            give, receive = Resource(normalized["give"]), Resource(normalized["receive"])
            self.players[seat].resources[give] -= self.trade_rate(seat, give)
            self.players[seat].resources[receive] += 1
            self.last_event = f"{self.players[seat].name} traded with the bank"
        elif kind == "end_turn":
            self.played_development = False
            self.current = (self.current + 1) % 4
            self.rolled = None
            self.phase = "roll"
            self.turn += 1
            self.last_event = f"{self.players[self.current].name}'s turn"
        if kind in {"build_road", "build_settlement"}:
            self._update_longest_road()
        turn_owner = self.robber_actor if self.phase == "discard" else self.current
        if turn_owner is not None:
            self._check_winner(turn_owner)

    def _offer_trade(self, seat: int, action: dict) -> None:
        if self.winner is not None or self.current != seat or self.phase != "actions":
            raise IllegalAction("trades require the active player's action phase")
        if set(action) != {"type", "recipient", "give", "receive"}:
            raise IllegalAction("trade requires recipient, give, and receive")
        recipient = action["recipient"]
        if type(recipient) is not int or recipient not in range(4) or recipient == seat:
            raise IllegalAction("invalid trade recipient")
        for field_name in ("give", "receive"):
            bundle = action[field_name]
            if not isinstance(bundle, dict) or not bundle:
                raise IllegalAction("each side must contain resources")
            if any(r not in {v.value for v in RESOURCES} or type(n) is not int or not 1 <= n <= 19 for r, n in bundle.items()):
                raise IllegalAction("invalid resource bundle")
        if set(action["give"]) & set(action["receive"]):
            raise IllegalAction("a resource cannot appear on both sides")
        if any(self.players[seat].resources[Resource(r)] < n for r, n in action["give"].items()):
            raise IllegalAction("offered cards are not held")
        self.trade_offer = {"owner": seat, "recipient": recipient, "give": dict(action["give"]), "receive": dict(action["receive"])}
        self.phase = "trade_response"
        self.current = recipient
        self.last_event = f"{self.players[seat].name} offered a trade to {self.players[recipient].name}"

    def _legal_settlements(self, seat: int, setup: bool = False) -> list[int]:
        if len(self.players[seat].settlements) >= 5:
            return []
        occupied = self.occupied()
        available = [v for v in range(len(self.board.vertices)) if v not in occupied and not any(n in occupied for n in self.board.neighbors[v])]
        if setup:
            return available
        connected = {v for edge in self.players[seat].roads for v in edge}
        return [v for v in available if v in connected]

    def _legal_roads(self, seat: int) -> list[tuple[int, int]]:
        if len(self.players[seat].roads) >= 15:
            return []
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
        if self.phase == "free_roads":
            p.roads.add(edge)
            self.free_roads -= 1
            if not self.free_roads or not self._legal_roads(seat):
                self.free_roads = 0
                self.phase = self.return_phase
            return
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
            if vertex in tile.vertices and isinstance(tile.resource, Resource) and self.bank[tile.resource] > 0:
                self.players[seat].resources[tile.resource] += 1

    def _roll(self) -> None:
        self.rolled = self.rng.randint(1, 6) + self.rng.randint(1, 6)
        if self.rolled == 7:
            self.return_phase = "actions"
            self.robber_actor = self.current
            self.discard_remaining = {
                seat: sum(player.resources.values()) // 2
                for seat, player in enumerate(self.players)
                if sum(player.resources.values()) > 7
            }
            self.phase = "discard" if self.discard_remaining else "robber"
            if self.discard_remaining:
                self.current = next(iter(self.discard_remaining))
            self.last_event = "A 7 was rolled; discard then move the robber"
            return
        occupied = self.occupied()
        claims = {r: [0] * 4 for r in RESOURCES}
        for i, tile in enumerate(self.board.tiles):
            if i == self.robber_tile or tile.number != self.rolled or not isinstance(tile.resource, Resource):
                continue
            for vertex in tile.vertices:
                if vertex in occupied:
                    owner = occupied[vertex]
                    amount = 2 if vertex in self.players[owner].cities else 1
                    claims[tile.resource][owner] += amount
        for resource, amounts in claims.items():
            if sum(amounts) <= self.bank[resource]:
                for player, amount in zip(self.players, amounts):
                    player.resources[resource] += amount
            elif sum(amount > 0 for amount in amounts) == 1:
                owner = next(i for i, amount in enumerate(amounts) if amount)
                self.players[owner].resources[resource] += self.bank[resource]
        self.phase = "actions"
        self.last_event = f"{self.players[self.current].name} rolled {self.rolled}"

    def _move_robber(self, seat: int, tile_index: int) -> None:
        self.robber_tile = tile_index
        self.phase = "steal" if self._robber_victims(seat) else self.return_phase
        self.last_event = f"{self.players[seat].name} moved the robber"

    def _robber_victims(self, seat: int) -> list[int]:
        occupied = self.occupied()
        adjacent = {occupied[v] for v in self.board.tiles[self.robber_tile].vertices if v in occupied}
        return sorted(s for s in adjacent if s != seat and sum(self.players[s].resources.values()))

    def _check_winner(self, seat: int) -> None:
        if not self.phase.startswith("setup") and self.players[seat].points >= 10:
            self.winner = seat
            self.phase = "finished"
            self.last_event = f"{self.players[seat].name} wins"

    def _update_longest_road(self) -> None:
        occupied = self.occupied()
        self.road_lengths = [
            longest_road(p.roads, {v for v, owner in occupied.items() if owner != seat})
            for seat, p in enumerate(self.players)
        ]
        self.longest_road_holder = award_holder(self.road_lengths, 5, self.longest_road_holder)
        for seat, player in enumerate(self.players):
            player.longest_road_award = seat == self.longest_road_holder

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
            "discard_remaining": self.discard_remaining.get(viewer, 0),
            "longest_road_holder": self.longest_road_holder,
            "trade_offer": self.trade_offer,
            "board": {
                "vertices": self.board.vertices,
                "edges": self.board.edges,
                "ports": [{"vertices": vertices, "resource": resource} for vertices, resource in self.board.ports],
                "tiles": [
                    {"q": t.q, "r": t.r, "resource": t.resource, "number": t.number, "vertices": t.vertices}
                    for t in self.board.tiles
                ],
            },
            "players": [
                {
                    "seat": i,
                    "name": p.name,
                    "points": p.points if i == viewer or self.winner is not None else p.public_points,
                    "development": [card for card, _ in p.development] if i == viewer else None,
                    "development_count": len(p.development),
                    "knights": p.knights,
                    "largest_army": p.largest_army,
                    "road_length": self.road_lengths[i],
                    "longest_road": p.longest_road_award,
                    "resource_count": sum(p.resources.values()),
                    "resources": ({r.value: n for r, n in p.resources.items()} if i == viewer else None),
                    "roads": sorted(p.roads),
                    "settlements": sorted(p.settlements),
                    "cities": sorted(p.cities),
                    "trade_rates": {r.value: self.trade_rate(i, r) for r in RESOURCES},
                }
                for i, p in enumerate(self.players)
            ],
            "legal_actions": legal,
            "events": self.observed_events(viewer),
        }
