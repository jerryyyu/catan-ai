"""Read-only verification of a solo pilot; actor labels do not prove human play."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from catan_ai.bots import make_policy
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation


def audit(record):
    if record['schema'] != 2 or record['source'] != source_files():
        raise ValueError('run this audit with the solo match frozen source')
    human = record['human_seat']
    if type(human) is not int or human not in range(4):
        raise ValueError('invalid human seat')
    if record['opponent'] not in ('heuristic', 'tactical', 'planner_parameter', 'planner_devfix'):
        raise ValueError('opponent does not support deterministic solo replay')
    bots = {seat: make_policy(record['opponent'], record['seed'] * 31 + seat)
            for seat in range(4) if seat != human}
    if record['opponent'] in ('planner_parameter', 'planner_devfix') and (
            not record.get('model_sha256') or any(bot.model_hash != record['model_sha256'] for bot in bots.values())):
        raise ValueError('saved checkpoint unavailable or changed')
    game = Game.new(record['names'], record['seed'])
    bot_times = []
    human_actions = 0
    turn_times = {}
    for row in record['trace']:
        seat = row['seat']
        if type(seat) is not int or seat not in range(4):
            raise ValueError('invalid action seat')
        actor = 'human' if seat == human else 'bot'
        if row['actor'] != actor:
            raise ValueError('actor label differs from registered solo seat')
        seconds = row['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid recorded action time')
        if actor == 'bot':
            chosen = bots[seat].choose(Observation.for_policy(game, seat, bots[seat]), seat)
            if chosen != row['action']:
                raise ValueError('bot decision differs from frozen replay')
            bot_times.append(seconds)
            if not game.phase.startswith('setup'):
                key = (seat, game.turn)
                turn_times[key] = turn_times.get(key, 0) + seconds
        else:
            human_actions += 1
        game.apply(seat, row['action'])
    if game.winner != record['winner'] or game.turn != record['turn']:
        raise ValueError('terminal metadata differs from replay')
    completed = game.winner is not None
    return {'replay_valid': True, 'completed': completed, 'opponent': record['opponent'],
            'human_seat': human, 'human_won': game.winner == human if completed else None,
            'recorded_human_actions': human_actions, 'recorded_bot_actions': len(bot_times),
            'maximum_recorded_bot_step_seconds': max(bot_times, default=None),
            'maximum_recorded_bot_turn_seconds': max(turn_times.values(), default=None),
            'recorded_step_budget_violations': sum(t > 2 for t in bot_times),
            'recorded_turn_budget_violations': sum(t > 10 for t in turn_times.values()),
            'human_participation_verified': False, 'experience_verified': False,
            'claim_limit': 'Solo usability pilot only. Actor labels and recorded durations are not '
                           'independent evidence of human participation, experience or clock accuracy. '
                           'Human timings exclude thinking time. Partial-turn totals may be incomplete. '
                           'No three-human transfer or randomized strength estimate is inferred.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.record.read_bytes()
    result = audit(json.loads(raw))
    result['record_sha256'] = hashlib.sha256(raw).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
