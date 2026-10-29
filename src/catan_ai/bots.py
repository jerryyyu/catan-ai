from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .engine import COSTS, Game, Resource


class Policy(Protocol):
    def choose(self, game: Game, seat: int) -> dict: ...


@dataclass
class RandomBot:
    rng: random.Random

    def choose(self, game: Game, seat: int) -> dict:
        legal = game.legal_actions(seat)
        if not legal:
            raise RuntimeError(f"no legal action for seat {seat}")
        return self.rng.choice(legal)


@dataclass
class HeuristicBot:
    rng: random.Random

    def choose(self, game: Game, seat: int) -> dict:
        legal = game.legal_actions(seat)
        if not legal:
            raise RuntimeError(f"no legal action for seat {seat}")
        by_type: dict[str, list[dict]] = {}
        for action in legal:
            by_type.setdefault(action["type"], []).append(action)

        if "roll" in by_type:
            return by_type["roll"][0]
        if "move_robber" in by_type:
            return max(by_type["move_robber"], key=lambda a: self._robber_score(game, seat, a["tile"]))
        if "build_city" in by_type:
            return max(by_type["build_city"], key=lambda a: self._vertex_score(game, a["vertex"]))
        if "build_settlement" in by_type:
            return max(by_type["build_settlement"], key=lambda a: self._vertex_score(game, a["vertex"]))
        if "build_road" in by_type and "end_turn" not in by_type:
            return max(by_type["build_road"], key=lambda a: self._road_score(game, seat, a["edge"]))
        if "build_road" in by_type and len(game.players[seat].roads) < 8 and self.rng.random() < 0.6:
            return max(by_type["build_road"], key=lambda a: self._road_score(game, seat, a["edge"]))
        if "bank_trade" in by_type:
            trade = self._useful_trade(game, seat, by_type["bank_trade"])
            if trade is not None:
                return trade
        return by_type["end_turn"][0]

    @staticmethod
    def _useful_trade(game: Game, seat: int, trades: list[dict]) -> dict | None:
        player = game.players[seat]
        targets = ["city", "settlement", "road"]
        for target in targets:
            if target == "city" and not player.settlements:
                continue
            missing = [resource for resource, amount in COSTS[target].items() if player.resources[resource] < amount]
            if not missing:
                continue
            for action in trades:
                give, receive = Resource(action["give"]), Resource(action["receive"])
                if receive in missing and player.resources[give] - COSTS[target].get(give, 0) >= 4:
                    return action
        return None

    @staticmethod
    def _vertex_score(game: Game, vertex: int) -> float:
        weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1, None: 0}
        adjacent = [tile for tile in game.board.tiles if vertex in tile.vertices]
        production = sum(weights[t.number] for t in adjacent)
        diversity = len({t.resource for t in adjacent if isinstance(t.resource, Resource)})
        return production + diversity * 1.5

    @classmethod
    def _road_score(cls, game: Game, seat: int, edge: list[int]) -> float:
        occupied = game.occupied()
        return max(
            cls._vertex_score(game, vertex) if vertex not in occupied else 0
            for vertex in edge
        )

    @staticmethod
    def _robber_score(game: Game, seat: int, tile_index: int) -> float:
        owners = game.occupied()
        score = 0.0
        for vertex in game.board.tiles[tile_index].vertices:
            owner = owners.get(vertex)
            if owner is not None and owner != seat:
                score += game.players[owner].points + 0.1 * sum(game.players[owner].resources.values())
        return score


def make_policy(name: str, seed: int) -> Policy:
    rng = random.Random(seed)
    if name == "random":
        return RandomBot(rng)
    if name == "heuristic":
        return HeuristicBot(rng)
    raise ValueError(f"unknown policy {name!r}; choose random or heuristic")
