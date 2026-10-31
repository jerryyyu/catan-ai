from catan_ai.belief import BeliefTracker
from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.guided_belief import GuidedBeliefTracker
from catan_ai.observation import Observation
from catan_ai.resource_constraints import resource_upper_before


def test_retrospective_bounds_never_exclude_the_generating_resource_history(monkeypatch):
    # The oracle uses simulator truth only in this test, never in the tracker.
    monkeypatch.setattr(BeliefTracker, 'update', lambda self, view: None)
    kinds = set()
    for seed in (1449200, 1449201):
        game = Game.new(['a', 'b', 'c', 'd'], seed)
        policies = [make_policy(name, seed * 17 + seat + 1)
                    for seat, name in enumerate(('tactical', 'builder', 'planner_robber', 'expander'))]
        history = []
        for step in range(1200):
            if game.winner is not None:
                break
            seat = game.current
            action = policies[seat].choose(Observation.for_policy(game, seat, policies[seat]), seat)
            kinds.add(action['type'])
            game.apply(seat, action)
            history.append([dict(player.resources) for player in game.players])
            if step % 50 and game.winner is None:
                continue
            for observer in range(4):
                view = Observation.from_game(game, observer)
                assert len(view.events) == len(history)
                tracker = GuidedBeliefTracker(0, 1)
                tracker.update(view)
                upper = resource_upper_before(view.events)
                for index, (event, hands) in enumerate(zip(view.events, history, strict=True)):
                    if index:
                        for owner, hand in enumerate(history[index - 1]):
                            assert all(n <= upper[id(event)][owner][r] for r, n in hand.items())
                    for owner, group, bound in tracker.subset_after[id(event)]:
                        assert sum(hands[owner][r] for r in group) >= bound
                    for owners, resource, bound in tracker.group_after[id(event)]:
                        assert sum(hands[owner][resource] for owner in owners) >= bound
        assert game.winner is not None
    assert {'discard', 'steal', 'bank_trade', 'buy_development'} <= kinds
