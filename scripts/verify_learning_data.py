"""Replay generated games and verify every retained observation/label boundary."""
import argparse
import hashlib
import json
from pathlib import Path

from catan_ai.engine import Game
from catan_ai.learning import action_features, state_features
from catan_ai.learn_pipeline import producer_source, SPLITS
from catan_ai.observation import Observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    games = examples = 0
    for split, (low, high) in SPLITS.items():
        for path in sorted((args.data / split).glob('*.json')):
            shard = json.loads(path.read_text())
            identity = shard['identity']
            seed = identity['seed']
            assert identity['source'] == producer_source()
            assert identity['split'] == split and low <= seed <= high
            names = identity['policies']
            names = names[seed % 4:] + names[:seed % 4]
            game = Game.new(names, seed)
            rows = {row['step']: row for row in shard['rows']}
            assert len(rows) == len(shard['rows'])
            trace = hashlib.sha256()
            for step, item in enumerate(shard['trajectory']):
                seat, action = item['seat'], item['action']
                assert seat == game.current and game.winner is None
                if step in rows:
                    row = rows.pop(step)
                    view = Observation.from_game(game, seat)
                    assert row['seat'] == seat and row['turn'] == game.turn
                    assert row['state'] == state_features(view)
                    assert all(a in view.actions for a in row['actions'])
                    assert row['features'] == [action_features(view, a) for a in row['actions']]
                    assert row['win'] == int(shard['winner'] == seat)
                    examples += 1
                trace.update(json.dumps([seat, action], sort_keys=True).encode())
                game.apply(seat, action)
            assert not rows
            assert shard['complete'] and game.winner == shard['winner'] and game.winner is not None
            assert trace.hexdigest() == shard['trace_sha256']
            games += 1
    assert games and examples
    print(json.dumps({'games_replayed': games, 'observation_rows_verified': examples}))


if __name__ == '__main__':
    main()
