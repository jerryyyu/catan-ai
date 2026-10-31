"""Fixed, deterministic opponent styles for robustness evaluation."""
from .bots import HeuristicBot
from .engine import COSTS, Resource


class StyleBot(HeuristicBot):
    def __init__(self, style):
        self.style = style

    def _placement(self, view, vertex):
        weights = ({Resource.ORE: 2, Resource.GRAIN: 1.6, Resource.WOOL: 1.2}
                   if self.style == 'builder' else {Resource.BRICK: 1.7, Resource.LUMBER: 1.7})
        return sum((6 - abs(7 - t.number)) * weights.get(t.resource, 1)
                   for t in view.board.tiles if vertex in t.vertices and t.number is not None)

    def choose(self, view, seat):
        legal = view.legal_actions(seat)
        groups = {}
        for action in legal:
            groups.setdefault(action['type'], []).append(action)
        if 'reject_trade' in groups:
            return groups['reject_trade'][0]
        for kind in ('roll', 'discard', 'steal', 'move_robber'):
            if kind in groups:
                keys = {'roll': lambda a: 0,
                        'discard': lambda a: view.players[seat].resources[Resource(a['resource'])],
                        'steal': lambda a: view.players[a['victim']].public_points,
                        'move_robber': lambda a: self._robber_score(view, seat, a['tile'])}
                return max(groups[kind], key=keys[kind])
        if 'play_development' in groups:
            return groups['play_development'][0]
        order = (('build_city', 'buy_development', 'build_settlement') if self.style == 'builder'
                 else ('build_settlement', 'build_city'))
        for kind in order:
            if kind in groups:
                return max(groups[kind], key=lambda a: self._placement(view, a['vertex']) if 'vertex' in a else 0)
        if 'build_road' in groups and ('end_turn' not in groups or
                len(view.players[seat].roads) < 12):
            return max(groups['build_road'], key=lambda a: self._road_score(view, seat, a['edge']))
        targets = ('city', 'development', 'settlement', 'road') if self.style == 'builder' else ('settlement', 'road', 'city')
        hand = view.players[seat].resources
        for target in targets:
            if target == 'city' and not view.players[seat].settlements:
                continue
            cost = COSTS[target]
            for action in groups.get('bank_trade', []):
                give, receive = Resource(action['give']), Resource(action['receive'])
                if hand[receive] < cost.get(receive, 0) and hand[give] - view.rates[give] >= cost.get(give, 0):
                    return action
        if 'buy_development' in groups:
            return groups['buy_development'][0]
        return groups['end_turn'][0]
