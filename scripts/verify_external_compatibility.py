"""Regression witnesses for external search-model rules corrections."""
from catan_ai.catanatron_adapter import new_external_game
from catan_ai.engine import Game
from catan_ai.external_compatibility import install_rule_compatibility
from catan_ai.scoring import longest_road
from catanatron.models.player import Color
from catanatron.models.enums import SETTLEMENT
from catanatron.models import board as board_module, actions


install_rule_compatibility()
local = Game.new(['a', 'b', 'c', 'd'], 1400090)
external, mapping = new_external_game(local.board, 1400090)
a, b = local.board.edges[0]
x, y = mapping.vertices[a], mapping.vertices[b]
board = external.state.board
board.roads[x, y] = board.roads[y, x] = Color.RED
board.buildings[y] = (Color.BLUE, SETTLEMENT)
assert len(board_module.longest_acyclic_path(board, {x}, Color.RED)) == longest_road({(a, b)}, {b}) == 1
external.state.player_state['P1_ACTUAL_VICTORY_POINTS'] = 10
assert external.winning_color() is None
assert external.copy().winning_color() is None
external.state.current_turn_index = 1
assert external.winning_color() == Color.BLUE
for deck in ([0, 3, 3, 3, 3], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]):
    options = actions.year_of_plenty_possibilities(Color.RED, deck)
    assert all(len(action.value) == min(sum(deck), 2) for action in options)
    assert bool(options) == (sum(deck) > 0)
print('PASS: blocked road endpoint, on-turn victory including copied states, scarce-bank Year of Plenty')
