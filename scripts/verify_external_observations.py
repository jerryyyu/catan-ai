"""Probe observation-only external states along a complete four-player game."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from catan_ai.belief import BeliefTracker, BeliefFailure
from catan_ai.bots import make_policy
from catan_ai.catanatron_adapter import from_observation
from catan_ai.engine import Game
from catan_ai.experiments import source_files
from catan_ai.observation import Observation
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particles', type=int, default=128)
    args = parser.parse_args()
    game = Game.new(['a', 'b', 'c', 'd'], 1400050)
    bots = [make_policy('tactical', s) for s in range(4)]
    trackers = [BeliefTracker(42 + s, particles=args.particles) for s in range(4)]
    rows = []
    searches = 0
    for step in range(10000):
        if game.winner is not None:
            break
        seat = game.current
        view = Observation.from_game(game, seat)
        if view.phase in {'actions', 'roll', 'free_roads'}:
            row = {'step': step, 'turn': view.turn, 'seat': seat, 'phase': view.phase}
            started = time.perf_counter()
            try:
                tracker_before = deepcopy(trackers[seat])
                external, mapping = from_observation(view, trackers[seat], 123 + step)
                altered = deepcopy(game)
                altered.deck.reverse()
                for s, p in enumerate(altered.players):
                    if s != seat:
                        p.development = [('victory_point' if c != 'victory_point' else 'knight', t)
                                         for c, t in p.development]
                altered_view = Observation.from_game(altered, seat)
                assert altered_view == view, 'hidden-state perturbation changed observation'
                other, _ = from_observation(altered_view, tracker_before, 123 + step)
                assert external.state.player_state == other.state.player_state
                assert external.state.development_listdeck == other.state.development_listdeck
                assert external.playable_actions == other.playable_actions
                if searches < 12 and len(view.actions) > 1:
                    action = AlphaBetaPlayer(list(Color)[seat]).decide(external, external.playable_actions)
                    assert mapping.local_action(action, view.actions, seat, view.rates) in view.actions
                    searches += 1
                row['pass'] = True
            except (BeliefFailure, ValueError) as exc:
                row.update({'pass': False, 'error_type': type(exc).__name__, 'error': str(exc)})
            row['seconds'] = time.perf_counter() - started
            rows.append(row)
        game.apply(seat, bots[seat].choose(view, seat))
    assert game.winner is not None, 'diagnostic game truncated'
    try:
        from_observation(game, trackers[0], 0)
    except TypeError:
        pass
    else:
        raise AssertionError('adapter accepted full engine state')
    failures = [r for r in rows if not r['pass']]
    result = {'seed': 1400050, 'particles': args.particles,
              'winner': game.winner, 'turn': game.turn, 'source': source_files(),
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'observation-derived state parity and hidden-state invariance; not a matchup',
              'external_search_decisions': searches, 'pass': not failures, 'cases': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'cases': len(rows), 'search_decisions': searches, 'failures': failures,
                      'winner': game.winner, 'turn': game.turn}, indent=2))


if __name__ == '__main__':
    main()
