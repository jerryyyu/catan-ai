import random
from collections import Counter

from fastapi.testclient import TestClient

from catan_ai.api import Session, advance_bots, app, sessions
from catan_ai.bots import HeuristicBot, make_policy
from catan_ai.engine import Game, Resource
from catan_ai.observation import Observation


def test_bot_trade_pauses_for_human_and_resumes_after_rejection():
    game = Game.new(['human', 'bot', 'b', 'c'], 42)
    game.phase, game.current = 'actions', 1
    game.players[1].resources[Resource.BRICK] = 1
    class HumanPartner(random.Random):
        def choice(self, choices):
            return 0 if 0 in choices else choices[0]
    bot = HeuristicBot(HumanPartner(1))
    session = Session(game, 0, {1: bot, 2: HeuristicBot(random.Random(2)), 3: HeuristicBot(random.Random(3))})
    sessions['trade-test'] = session
    advance_bots(session)
    assert game.phase == 'trade_response' and game.current == 0
    client = TestClient(app)
    response = client.post('/api/games/trade-test/actions', json={'action': {'type': 'reject_trade'}})
    assert response.status_code == 200
    assert game.turn > 0
    assert bot.last_offer_turn == 0
    sessions.pop('trade-test')


def test_replay_and_conservation_across_full_games():
    for seed in range(4):
        game = Game.new(['a', 'b', 'c', 'd'], seed)
        replay = Game.new(['a', 'b', 'c', 'd'], seed)
        bots = [make_policy('heuristic', seed * 4 + s) for s in range(4)]
        counts = Counter()
        for _ in range(10000):
            if game.winner is not None:
                break
            seat = game.current
            action = bots[seat].choose(Observation.from_game(game, seat), seat)
            counts[action['type']] += 1
            game.apply(seat, action)
            replay.apply(seat, action)
            assert game.public_state(0) == replay.public_state(0)
            assert game.rng.getstate() == replay.rng.getstate()
            for resource in Resource:
                assert 0 <= game.bank[resource] <= 19
                assert all(p.resources[resource] >= 0 for p in game.players)
            assert all(len(p.roads) <= 15 and len(p.settlements) <= 5 and len(p.cities) <= 4 for p in game.players)
            assert all(game.public_state(0)['players'][s]['development'] is None for s in (1, 2, 3))
        assert game.winner is not None
        assert counts['offer_trade'] > 0
        assert counts['buy_development'] > 0
