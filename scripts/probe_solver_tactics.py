"""Compare resource-solver tactics with identical observed-history constraints."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import time

from catan_ai import constraint_belief as constraints
from catan_ai.belief import BeliefTracker
from probe_resource_solver import CAPTURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--one-hot', action='store_true')
    parser.add_argument('--resource-source', type=Path, default=Path(
        'runs/competitive-v10-source/src/catan_ai/constraint_belief.py'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = CAPTURES + ['external-solver-regression-v10', 'external-solver-regression-v11']
    module_source = args.resource_source.read_text()
    function = next(node for node in ast.parse(module_source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'solve_resources')
    source = ast.get_source_segment(module_source, function)
    constructors = [line for line in source.splitlines()
                    if line.strip().startswith('solver = z3.Solver')]
    assert len(constructors) == 1
    constructor = constructors[0].strip()
    plan = {'captures': names, 'variants': ['default', 'QF_LIA'] if args.one_hot else ['default', 'QF_LIA', 'QF_FD'],
            'one_hot': args.one_hot,
            'seeds': [0, 24719889, 24735019], 'solver_timeout_ms': 500,
            'maximum_full_witness_seconds': 1,
            'resource_producer_sha256': hashlib.sha256(source.encode()).hexdigest()}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    if args.one_hot:
        substitutions = {
            "choice = z3.Int(f'card_{index}')":
                "choice = [z3.Bool(f'card_{index}_{r}') for r in range(5)]",
            'solver.add(choice >= 0, choice < 5)':
                'solver.add(z3.PbEq([(b, 1) for b in choice], 1))',
            'z3.If(choice == r, 1, 0)': 'z3.If(choice[r], 1, 0)',
            'RESOURCES[model.eval(choice).as_long()].value':
                'RESOURCES[next(r for r, b in enumerate(choice) if z3.is_true(model.eval(b)))].value',
        }
        for old, new in substitutions.items():
            assert source.count(old) == 1
            source = source.replace(old, new)
    rows = []
    for variant in plan['variants']:
        namespace = dict(vars(constraints))
        replacement = 'solver = z3.Solver()' if variant == 'default' else (
            f'solver = z3.SolverFor({variant!r})')
        exec(compile(source.replace(constructor, replacement),
                     '<resource-tactic-probe>', 'exec'), namespace)
        solve = namespace['solve_resources']
        for name in names:
            data = (Path('runs') / name / 'observation.pkl').read_bytes()
            view = pickle.loads(data)
            for seed in plan['seeds']:
                started = time.perf_counter()
                error = None
                try:
                    resources = solve(view, seed=seed, timeout_ms=500)
                    development = constraints.solve_development(view, seed=seed, timeout_ms=500)
                    particle = constraints.verify_witness(view, resources, development)
                    tracker = BeliefTracker(0, 1)
                    tracker.population = [particle]
                    tracker.history, tracker.seat = list(view.events), view.seat
                    world = tracker.sample(view, 0)
                    assert world.legal_actions(view.seat) == list(view.actions)
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                elapsed = time.perf_counter() - started
                row = {'variant': variant, 'capture': name, 'seed': seed,
                       'input_sha256': hashlib.sha256(data).hexdigest(),
                       'seconds': elapsed, 'error': error,
                       'pass': error is None and elapsed <= 1}
                rows.append(row)
                with (args.output / 'rows.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
    (args.output / 'summary.json').write_text(json.dumps({'plan': plan, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
