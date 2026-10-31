from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar, Protocol

from .engine import COSTS, Resource
from .observation import Observation as Game


class Policy(Protocol):
    def choose(self, game: Game, seat: int) -> dict: ...


@dataclass
class RandomBot:
    needs_history: ClassVar[bool] = False
    rng: random.Random

    def choose(self, game: Game, seat: int) -> dict:
        legal = game.legal_actions(seat)
        if not legal:
            raise RuntimeError(f"no legal action for seat {seat}")
        return self.rng.choice(legal)


@dataclass
class HeuristicBot:
    needs_history: ClassVar[bool] = False
    rng: random.Random
    last_offer_turn: int = -1

    def choose(self, game: Game, seat: int) -> dict:
        legal = game.legal_actions(seat)
        if not legal:
            raise RuntimeError(f"no legal action for seat {seat}")
        by_type: dict[str, list[dict]] = {}
        for action in legal:
            by_type.setdefault(action["type"], []).append(action)
        if "reject_trade" in by_type:
            offer = game.trade_offer
            if "accept_trade" in by_type and sum(offer["give"].values()) >= sum(offer["receive"].values()):
                return by_type["accept_trade"][0]
            return by_type["reject_trade"][0]

        if "play_development" in by_type:
            return self.rng.choice(by_type["play_development"])

        if "roll" in by_type:
            return by_type["roll"][0]
        if "discard" in by_type:
            return max(by_type["discard"], key=lambda a: game.players[seat].resources[Resource(a["resource"])])
        if "steal" in by_type:
            return max(by_type["steal"], key=lambda a: game.players[a["victim"]].public_points)
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
        if "buy_development" in by_type:
            return by_type["buy_development"][0]
        if game.phase == "actions" and self.last_offer_turn != game.turn:
            offer = self._offer(game, seat)
            if offer is not None:
                self.last_offer_turn = game.turn
                return offer
        return by_type["end_turn"][0]

    def _offer(self, game: Game, seat: int) -> dict | None:
        hand = game.players[seat].resources
        for target in ("city", "settlement", "development"):
            cost = COSTS[target]
            missing = [r for r, n in cost.items() if hand[r] < n]
            surplus = [r for r in Resource if hand[r] > cost.get(r, 0)]
            if missing and surplus:
                recipient = self.rng.choice([s for s in range(4) if s != seat])
                return {"type": "offer_trade", "recipient": recipient,
                        "give": {surplus[0].value: 1}, "receive": {missing[0].value: 1}}
        return None

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
                if receive in missing and player.resources[give] - COSTS[target].get(give, 0) >= game.trade_rate(seat, give):
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
                score += game.players[owner].public_points + 0.1 * game.players[owner].resource_count
        return score


def make_policy(name: str, seed: int) -> Policy:
    rng = random.Random(seed)
    if name == 'planner_parameter':
        from .parameter_planner import ParameterPlanner
        return ParameterPlanner(rng)
    if name in ('planner_tactical_opening', 'tactical_planner_opening'):
        from .components import ComponentPolicy
        opening, turn = (('tactical', 'planner_robber') if name == 'planner_tactical_opening'
                         else ('planner_robber', 'tactical'))
        return ComponentPolicy(opening, turn, seed)
    if name == 'tactical_external_opening':
        from .components import ExternalOpeningTactical
        return ExternalOpeningTactical(rng, seed)
    if name == 'planner_external_opening':
        from .external_opening import ExternalOpeningPlanner
        return ExternalOpeningPlanner(rng, seed)
    if name == 'planner_finish':
        from .finish import FinishPlanner
        return FinishPlanner(rng)
    if name == 'planner_value_draw':
        from .learned_planner import ValueDrawPlanner
        return ValueDrawPlanner(rng)
    if name == 'planner_value':
        from .learned_planner import ValuePlanner
        return ValuePlanner(rng)
    if name == 'planner_chain':
        from .chain_planner import ChainPlanner
        return ChainPlanner(rng)
    if name == 'planner_cards':
        from .card_planner import CardPlanner
        return CardPlanner(rng)
    if name == 'planner_purchase_opening':
        from .opening import PurchaseOpeningPlanner
        return PurchaseOpeningPlanner(rng)
    if name == 'planner_opening':
        from .opening import OpeningPlanner
        return OpeningPlanner(rng)
    if name == 'planner':
        from .planner import EconomicPlanner
        return EconomicPlanner(rng)
    if name == 'planner_development':
        from .planner import DevelopmentPlanner
        return DevelopmentPlanner(rng)
    if name == 'planner_coverage':
        from .planner import CoveragePlanner
        return CoveragePlanner(rng)
    if name == 'planner_robber':
        from .planner import ProductionRobberPlanner
        return ProductionRobberPlanner(rng)
    if name == 'planner_offer':
        from .planner import OfferPlanner
        return OfferPlanner(rng)
    if name == 'planner_dev_sequence':
        from .planner import DevelopmentSequencePlanner
        return DevelopmentSequencePlanner(rng)
    if name == 'planner_card_timing':
        from .card_timing import CardTimingPlanner
        return CardTimingPlanner(rng)
    if name == 'external':
        from .external_policy import ExternalSearchBot
        return ExternalSearchBot(seed)
    if name == 'learned':
        import os
        from .learning import LearnedBot
        return LearnedBot(seed, os.environ.get('CATAN_MODEL_PATH'))
    if name in {'builder', 'expander'}:
        from .styles import StyleBot
        return StyleBot(name)
    if name in {'search_fast', 'search_sequence', 'search_alloc'}:
        from .search import MonteCarloBot, SearchConfig
        return MonteCarloBot(seed, SearchConfig(
            candidates=3, selection_samples=4 if name == 'search_alloc' else 2,
            confirmation_samples=5 if name == 'search_alloc' else 8,
            selection_rule='mean', root_tactical=True,
            sequence_candidates=name == 'search_sequence'))
    if name == "random":
        return RandomBot(rng)
    if name == "heuristic":
        return HeuristicBot(rng)
    if name == "strategic":
        from .strategic import StrategicBot
        return StrategicBot(rng)
    if name == "mc":
        from .search import MonteCarloBot
        return MonteCarloBot(seed)
    if name == 'tactical':
        from .tactical import TacticalBot
        return TacticalBot(rng)
    if name in {'mc_mean', 'mc_tactical'}:
        from .search import MonteCarloBot, SearchConfig
        return MonteCarloBot(seed, SearchConfig(selection_rule='mean',
                                              tactical=name == 'mc_tactical'))
    raise ValueError(f"unknown policy {name!r}; choose random, heuristic, strategic, or mc")
