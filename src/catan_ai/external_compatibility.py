"""Explicit rules corrections applied inside the external search environment.

The pinned third-party files remain unchanged. Search/evaluation stay external;
these shims align road endpoints, victory timing and mandatory card selection.
"""


def install_rule_compatibility():
    from catanatron.game import Game
    from catanatron.models import board as board_module, actions
    if getattr(Game, '_catan_ai_rules_installed', False):
        return

    def winning_color(game):
        state = game.state
        seat = state.current_turn_index
        return state.colors[seat] if state.player_state[f'P{seat}_ACTUAL_VICTORY_POINTS'] >= game.vps_to_win else None

    def longest_path(board, node_set, color):
        best = []
        def visit(vertex, path, used):
            nonlocal best
            if len(path) > len(best):
                best = path
            if path and board.is_enemy_node(vertex, color):
                return
            for neighbor in board_module.STATIC_GRAPH.neighbors(vertex):
                edge = tuple(sorted((vertex, neighbor)))
                if edge not in used and board.is_friendly_road(edge, color):
                    visit(neighbor, path + [edge], used | {edge})
        for vertex in node_set:
            visit(vertex, [], set())
        return best

    original_plenty = actions.year_of_plenty_possibilities
    def plenty(color, deck):
        options = original_plenty(color, deck)
        count = min(sum(deck), 2)
        return [action for action in options if len(action.value) == count]

    Game.winning_color = winning_color
    board_module.longest_acyclic_path = longest_path
    actions.year_of_plenty_possibilities = plenty
    Game._catan_ai_rules_installed = True
