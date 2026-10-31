"""Local browser-test server; fixture endpoints are absent from production."""
import os
from tempfile import TemporaryDirectory

from fastapi import FastAPI
from catan_ai.api import app as production, create_game, NewGame, sessions
from catan_ai.engine import Resource

_records = TemporaryDirectory(prefix='catan-browser-tests-')
os.environ['CATAN_PILOT_DIR'] = _records.name

app = FastAPI()


@app.post('/fixtures/{scenario}')
def fixture(scenario: str):
    result = create_game(NewGame(name='You', seed=71))
    game = sessions[result['game_id']].game
    game.phase, game.turn = 'actions', 4
    game.players[0].settlements.add(0)
    game.players[0].resources = {r: 3 for r in Resource}
    game.players[1].resources = {r: 2 for r in Resource}
    if scenario in ('knight', 'monopoly', 'year_of_plenty', 'road_building'):
        game.players[0].development = [(scenario, 0)]
    elif scenario == 'discard':
        game.phase = 'discard'
        game.discard_remaining = {0: 2}
        game.robber_actor = 0
    elif scenario == 'victim':
        game.phase = 'steal'
        game.players[1].settlements.add(game.board.tiles[game.robber_tile].vertices[2])
    elif scenario == 'port':
        edge = next(edge for edge, kind in game.board.ports if kind == Resource.BRICK)
        game.players[0].settlements.add(edge[0])
    elif scenario == 'trade_response':
        game.current = 1
        game.apply(1, {'type': 'offer_trade', 'recipient': 0, 'give': {'ore': 1}, 'receive': {'brick': 1}})
    elif scenario == 'win':
        game.players[0].development = [('victory_point', 0)] * 5
        game.players[0].cities = {20, 30}
    result['state'] = game.public_state(0)
    return result


app.mount('/', production)
