"""Require all predeclared comparisons to pass simultaneous strength bounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from catan_ai.inference import paired_normal_lcb
from catan_ai.search_eval import summarize
from catan_ai.experiments import source_files


def evaluate(plan, results):
    entries = plan['runs']
    if not entries:
        raise ValueError('every planned run is required')
    count = sum(len(entry['config']['panels']) for entry in entries)
    if count != plan['comparison_count'] or not 0 < plan['family_alpha'] <= .05:
        raise ValueError('invalid predeclared comparison family')
    candidates = {entry['config']['candidate'] for entry in entries}
    controls = {entry['config']['control'] for entry in entries}
    if len(candidates) != 1 or controls != {'tactical', 'external'}:
        raise ValueError('one challenger must face both required controls')
    alpha = plan['family_alpha'] / count
    panels = []
    used_seeds = set()
    results = iter(results)
    for entry in entries:
        try:
            result = next(results)
        except StopIteration:
            raise ValueError('every planned run is required') from None
        config = entry['config']
        if result['config'] != config or result['source'] != plan['source']:
            raise ValueError('result does not match the frozen config/source')
        if result.get('checkpoint') != plan.get('checkpoint'):
            raise ValueError('checkpoint mismatch')
        if config['stage'] != 'heldout' or not config.get('requires_dev_pass'):
            raise ValueError('only admitted held-out runs are confirmatory')
        gates = config['gates']
        if (gates.get('maximum_match_truncations') != 0
                or gates.get('maximum_step_seconds', float('inf')) > 2
                or gates.get('maximum_turn_seconds', float('inf')) > 10):
            raise ValueError('required completion and time gates missing')
        for panel in config['panels']:
            seeds = set(range(panel['seed'], panel['seed'] + config['seeds_per_panel']))
            if used_seeds & seeds:
                raise ValueError('confirmatory panels must use distinct board seeds')
            used_seeds.update(seeds)
        for row in result['games']:
            for arm in ('candidate', 'baseline'):
                game = row[arm]
                panel_config = next(p for p in config['panels'] if p['name'] == row['panel'])
                names = list(panel_config['opponents'])
                names.insert(row['seat'], config['candidate'] if arm == 'candidate' else config['control'])
                learned = {'planner_parameter', 'planner_value', 'planner_value_draw', 'learned'}
                expected_models = {str(seat) for seat, name in enumerate(names) if name in learned}
                metadata = game.get('policy_metadata', {})
                if set(metadata) != expected_models:
                    raise ValueError('per-game learned-policy coverage mismatch')
                for evidence in metadata.values():
                    checkpoint = plan.get('checkpoint')
                    if checkpoint is None or evidence != {
                            'model_sha256': checkpoint['sha256'], 'load_error': None}:
                        raise ValueError('per-game checkpoint mismatch or fallback')
                clocks = [t for key in ('decision_seconds', 'step_seconds')
                          for seat_times in game.get(key, []) for t in seat_times]
                clocks += [turn['seconds'] for turn in game.get('turn_timings', [])]
                clocks += [d['seconds'] for evidence in game.get('external', {}).values()
                           for d in evidence['diagnostics']]
                if any(isinstance(t, bool) or not isinstance(t, (int, float))
                       or not math.isfinite(t) or t < 0 for t in clocks):
                    raise ValueError('timings must be finite nonnegative seconds')
        summary = summarize(config, result['games'])
        if summary != result['summary']:
            raise ValueError('stored summary disagrees with raw games')
        for name, panel in summary['panels'].items():
            valid = panel['match_truncations'] == 0 and panel['external_faults'] == 0
            lcb = paired_normal_lcb(panel['seed_cluster_differences'], alpha) if valid else None
            panels.append({'control': config['control'], 'panel': name,
                           'paired_win_lift': panel['paired_win_lift'],
                           'simultaneous_lcb': lcb, 'alpha': alpha,
                           'pass': bool(panel['pass'] and lcb is not None and lcb > 0)})
        del result
    if next(results, None) is not None:
        raise ValueError('every planned run is required')
    return {'panels': panels, 'pass': all(row['pass'] for row in panels),
            'inference': 'approximate paired normal over board clusters; Bonferroni family'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if source_files() != plan['source']:
        raise ValueError('reader dependencies differ from the frozen source')
    hashes = []

    def results():
        for entry in plan['runs']:
            path = Path(entry['result'])
            with path.open('rb') as stream:
                hashes.append(hashlib.file_digest(stream, 'sha256').hexdigest())
            with path.open() as stream:
                yield json.load(stream)

    result = evaluate(plan, results())
    result.update(plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), result_sha256=hashes,
                  reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
