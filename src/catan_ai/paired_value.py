"""Public action representations for paired continuation-value learning."""
from copy import deepcopy

from .learning import action_features
from .observation import Observation
from .tactical import public_projection
from .value_model import value_features


def supported(action):
    return action['type'] in ('build_city', 'build_settlement', 'build_road',
                              'bank_trade', 'buy_development', 'end_turn')


def representation(view, action, planner):
    if action not in view.actions or not supported(action):
        raise ValueError('representation needs a supported legal action')
    root = public_projection(view)
    child = deepcopy(root, {id(root.board): root.board})
    purchase = action['type'] == 'buy_development'
    if purchase:
        child.players[view.seat].pay('development')
    else:
        child.apply(view.seat, action)
    before = value_features(view)
    after = value_features(Observation.from_game(child, view.seat, include_events=False))
    delta = [b - a for a, b in zip(before, after, strict=True)]
    action_vector = action_features(view, action)
    own = view.players[view.seat]
    context = [(own.public_points + own.development.count('victory_point')) / 10,
               max(p.public_points for s, p in enumerate(view.players) if s != view.seat) / 10,
               sum(before[13:18])]
    score = planner._value(child, view.seat) - planner._value(root, view.seat)
    score += planner.purchase_value if purchase else 0
    features = action_vector + delta + [x * c for c in context for x in action_vector] + [score / 100]
    return features, score


def shortlist(view, baseline, planner, limit=4):
    if not supported(baseline):
        return []
    candidates = []
    for action in view.actions:
        if supported(action):
            features, score = representation(view, action, planner)
            candidates.append({'action': action, 'features': features, 'immediate_score': score})
    candidates.sort(key=lambda row: row['immediate_score'], reverse=True)
    chosen = [next(row for row in candidates if row['action'] == baseline)]
    kinds = {baseline['type']}
    for row in candidates:
        if len(chosen) == limit:
            break
        if row['action']['type'] not in kinds:
            chosen.append(row)
            kinds.add(row['action']['type'])
    for row in candidates:
        if len(chosen) == limit:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen
