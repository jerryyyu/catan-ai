"""What the screen reader must return: a board position in screen terms.

Positions are given the way a person reads a pointy-top hex board: five rows
of 3, 4, 5, 4, 3 hexes, top to bottom, each counted left to right from 0. A
corner or side is named relative to one adjacent hex. Any adjacent hex may be
used; the reconstruction resolves all of them to the same engine vertex or edge.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Terrain = Literal['brick', 'lumber', 'wool', 'grain', 'ore', 'desert']
Corner = Literal['top', 'top_right', 'bottom_right', 'bottom', 'bottom_left', 'top_left']
Side = Literal['top_right', 'right', 'bottom_right', 'bottom_left', 'left', 'top_left']
Phase = Literal['setup', 'roll', 'actions', 'move_robber', 'steal', 'discard', 'trade_response', 'other_player']


class Hex(BaseModel):
    row: int = Field(ge=0, le=4)
    col: int = Field(ge=0, le=4)


class Tile(Hex):
    terrain: Terrain
    number: Optional[int] = Field(default=None, description='Number token; null for the desert')


class Building(Hex):
    color: str
    kind: Literal['settlement', 'city']
    corner: Corner


class Road(Hex):
    color: str
    side: Side


class Port(Hex):
    side: Side = Field(description='Coastal side of this hex that the port touches')
    resource: Optional[Literal['brick', 'lumber', 'wool', 'grain', 'ore']] = Field(
        default=None, description='null for a generic 3:1 port')


class Player(BaseModel):
    color: str
    name: str
    is_me: bool
    visible_points: int
    resource_cards: int
    development_cards: int
    knights_played: int
    longest_road: bool
    largest_army: bool


class Hand(BaseModel):
    brick: int = 0
    lumber: int = 0
    wool: int = 0
    grain: int = 0
    ore: int = 0


class Offer(BaseModel):
    from_color: str
    gives_me: Hand
    wants_from_me: Hand


class Position(BaseModel):
    tiles: list[Tile]
    robber: Hex
    ports: list[Port]
    buildings: list[Building]
    roads: list[Road]
    players: list[Player] = Field(description='In turn order, starting with any player')
    my_hand: Hand
    my_development_cards: list[Literal['knight', 'victory_point', 'monopoly', 'year_of_plenty', 'road_building']]
    current_color: str = Field(description='Color of the player whose turn it is')
    phase: Phase
    trade_offer: Optional[Offer] = Field(default=None, description='Only when another player is offering me a trade')
    dice: Optional[int] = Field(default=None, description='Total shown on the dice this turn, if rolled')
    uncertain: list[str] = Field(description='Anything that could not be read reliably')
