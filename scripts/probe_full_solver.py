"""Verify complete constraint witnesses through existing belief/world checks."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import z3
from catan_ai.belief import BeliefTracker
from catan_ai.constraint_belief import solve_particle
from catan_ai.experiments import source_files
from probe_resource_solver import CAPTURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in CAPTURES:
        data = (Path('runs') / name / 'observation.pkl').read_bytes()
        view = pickle.loads(data)
        started = time.perf_counter()
        error = None
        try:
            particle = solve_particle(view)
            tracker = BeliefTracker(0, 1)
            tracker.population, tracker.history, tracker.seat = [particle], list(view.events), view.seat
            world = tracker.sample(view, 0)
            if world.legal_actions(view.seat) != list(view.actions):
                raise RuntimeError('world legal actions differ from observation')
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        elapsed = time.perf_counter() - started
        row = {'capture': name, 'input_sha256': hashlib.sha256(data).hexdigest(),
               'seconds': elapsed, 'error': error, 'pass': error is None and elapsed <= 1}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'source': source_files(), 'solver_version': z3.get_version_string(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rows': rows, 'pass': all(row['pass'] for row in rows)}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
