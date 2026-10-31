"""Validate a private room replay; registration is not proof of human participation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from catan_ai.rooms import replay_bot
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


def audit(record):
    if record['schema'] != 1 or record['source'] != source_files():
        raise ValueError('run this audit with the room’s frozen source')
    seat = record['bot_seat']
    if type(seat) is not int or seat not in range(4):
        raise ValueError('invalid bot seat')
    expected = {str(s) for s in range(4) if s != seat}
    participants = record['participants']
    if set(participants) != expected:
        raise ValueError('three registered human seats required')
    for p in participants.values():
        if p['consent'] is not True or type(p['completed_matches']) is not int or p['completed_matches'] < 0:
            raise ValueError('invalid consent or experience registration')
    game = Game.new(record['names'], record['seed'])
    bot = replay_bot(record)
    times = []
    human_actions = 0
    for row in record['trace']:
        actor = 'bot' if row['seat'] == seat else 'human'
        if row['actor'] != actor:
            raise ValueError('action actor does not match registered seat')
        seconds = row['seconds']
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid recorded decision time')
        if actor == 'bot':
            action = bot.choose(Observation.for_policy(game, seat, bot), seat)
            if action != row['action']:
                raise ValueError('bot decision differs from frozen replay')
            times.append(seconds)
        else:
            human_actions += 1
        game.apply(row['seat'], row['action'])
    if game.winner != record['winner'] or game.turn != record['turn']:
        raise ValueError('terminal metadata differs from replay')
    return {'replay_valid': True, 'completed': game.winner is not None,
            'bot': record['bot'], 'bot_seat': seat,
            'bot_won': game.winner == seat if game.winner is not None else None,
            'registered_experienced_seats': sum(p['completed_matches'] >= 50 for p in participants.values()),
            'recorded_human_actions': human_actions, 'recorded_bot_actions': len(times),
            'maximum_recorded_bot_step_seconds': max(times, default=None),
            'human_participation_verified': False,
            'claim_limit': 'Pilot only. Registration and actor labels cannot establish actual human play. '
                           'No randomized comparison or turn-time coverage is inferred.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = args.record.read_bytes()
    result = audit(json.loads(data))
    result['record_sha256'] = hashlib.sha256(data).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
