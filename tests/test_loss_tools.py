import copy
import functools
import importlib.util
import json
from pathlib import Path
import random

import pytest

from catan_ai.sim import play_game

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attribution = load('loss_attribution')
replay = load('replay_swing_boards')
registry = load('build_results_registry')

LINEUP = ['builder', 'expander', 'builder']
CONFIG = {'candidate': 'heuristic', 'control': 'builder', 'max_turns': 400,
          'panels': [{'name': 'p', 'seed': 50, 'opponents': LINEUP}]}


def game_row(seed, seat, cand_winner, ctrl_winner, truncated=False, external=None):
    def arm(winner):
        return {'winner': None if truncated else winner, 'turns': 80, 'points': [10 if s == winner else 6 for s in range(4)],
                'truncated': truncated, 'external': external or {}}
    return {'panel': 'p', 'seed': seed, 'seat': seat, 'candidate': arm(cand_winner), 'baseline': arm(ctrl_winner)}


def test_points_at_reads_the_latest_value_not_the_peak():
    # Longest Road gained on turn 35 and lost on turn 39.
    history = {10: 3, 35: 7, 39: 5, 55: 6}
    assert replay.points_at(history, 40) == 5
    assert replay.points_at(history, 60) == 6
    assert replay.points_at(history, 5) == 0


def test_terminal_differences_catch_every_field_codex_injected():
    stored = {'winner': 2, 'turns': 81, 'points': [6, 7, 10, 5], 'truncated': False}
    assert replay.terminal_differences(stored, dict(stored)) == {}
    injected = {'winner': None, 'turns': 999, 'points': [0, 0, 0, 0], 'truncated': True}
    assert set(replay.terminal_differences(stored, injected)) == {'winner', 'turns', 'points', 'truncated'}
    # A different opponent winning, or only the points or turns changing, is still caught.
    assert set(replay.terminal_differences(stored, {**stored, 'winner': 1})) == {'winner'}
    assert set(replay.terminal_differences(stored, {**stored, 'points': [6, 8, 10, 5]})) == {'points'}
    assert set(replay.terminal_differences(stored, {**stored, 'turns': 82})) == {'turns'}


@functools.cache
def _real_report(seeds=(50, 51, 52)):
    """A v2 report built from real games, so replays must reproduce them exactly."""
    rows = []
    for seed in seeds:
        for seat in range(4):
            names = list(LINEUP)
            names.insert(seat, CONFIG['candidate'])
            cand = play_game(names, seed, CONFIG['max_turns']).as_dict()
            names[seat] = CONFIG['control']
            ctrl = play_game(names, seed, CONFIG['max_turns']).as_dict()
            rows.append({'panel': 'p', 'seed': seed, 'seat': seat, 'candidate': cand, 'baseline': ctrl})
    _, panels = attribution.analyse(iter(rows), 10, 7)
    return {'schema': 'loss-attribution/v2', 'receipt_sha256': 'x', 'config': CONFIG, 'source': {'a': '1'},
            'checkpoint': None, 'sampling': {}, 'receipt': 'r', 'panels': panels}


def real_report():
    return copy.deepcopy(_real_report())


def test_real_replays_reproduce_receipt_rows_and_tampering_is_excluded():
    report = real_report()
    selected = [r for block in report['panels']['p']['swing'].values() for r in block['selected']]
    if not selected:
        pytest.skip('no swing rows on these seeds')
    results, mismatches, summary = replay.run(report, 'p')
    assert mismatches == []
    assert sum(len(v) for v in results.values()) == len(selected)
    tampered = copy.deepcopy(report)
    for block in tampered['panels']['p']['swing'].values():
        for row in block['selected']:
            row['control']['turns'] += 1
    _, mismatches, _ = replay.run(tampered, 'p')
    assert len(mismatches) == len(selected)
    assert all(set(m['differences']) == {'control'} for m in mismatches)


def test_fail_open_injection_is_rejected_not_counted():
    report = real_report()
    if not any(b['selected'] for b in report['panels']['p']['swing'].values()):
        pytest.skip('no swing rows on these seeds')

    def corrupted(lineup, seed, max_turns, seat):
        track = replay.replay(lineup, seed, max_turns, seat)
        if lineup[seat] == CONFIG['candidate']:
            track['final'].update(winner=None, turns=999, points=[0, 0, 0, 0], truncated=True)
        return track
    results, mismatches, _ = replay.run(report, 'p', replay_fn=corrupted)
    assert all(not rows for rows in results.values())
    assert mismatches and all('candidate' in m['differences'] for m in mismatches)


def test_identity_binding_refuses_old_reports_and_drifted_producers():
    good = {'schema': 'loss-attribution/v2', 'receipt_sha256': 'x', 'config': CONFIG, 'source': {'a': '1'},
            'checkpoint': {'sha256': 'c'}}
    replay.check_identity(good, {'a': '1'}, {'sha256': 'c'})
    with pytest.raises(SystemExit, match='schema v2'):
        replay.check_identity({**good, 'schema': None}, {'a': '1'}, {'sha256': 'c'})
    with pytest.raises(SystemExit, match='source'):
        replay.check_identity(good, {'a': '2'}, {'sha256': 'c'})
    with pytest.raises(SystemExit, match='checkpoint'):
        replay.check_identity(good, {'a': '1'}, {'sha256': 'd'})
    with pytest.raises(SystemExit, match='receipt_sha256'):
        replay.check_identity({**good, 'receipt_sha256': None}, {'a': '1'}, {'sha256': 'c'})


def test_sampling_reaches_every_board_and_weights_by_eligible_seats():
    rows = [{'seed': seed, 'seat': seat} for seed in range(200) for seat in range(1 + seed % 3)]
    first = attribution.sample_swing(rows, 20, random.Random('s'))
    assert first == attribution.sample_swing(rows, 20, random.Random('s'))
    assert all(r['weight'] == 1 + r['seed'] % 3 for r in first)
    reached = set()
    for draw in range(300):
        reached.update(r['seed'] for r in attribution.sample_swing(rows, 5, random.Random(draw)))
    assert max(reached) >= 190 and len(reached) > 180  # later rows are not excluded by file order
    assert len(attribution.sample_swing(rows, 10 ** 6, random.Random(1))) == 200


def test_truncated_and_faulted_rows_are_excluded_and_counted():
    fault = {'1': {'diagnostics': [{'fallback': True, 'budget_failure': False, 'phase': 'actions', 'turn': 3,
                                    'seconds': .1}]}}
    rows = [game_row(1, 0, 0, 1), game_row(2, 0, None, None, truncated=True), game_row(3, 0, 0, 1, external=fault),
            game_row(4, 0, 1, 0)]
    _, panels = attribution.analyse(iter(rows), 10, 3)
    panel = panels['p']
    assert panel['excluded'] == {'truncated': 1, 'faulted': 1}
    assert [r['seed'] for r in panel['swing']['candidate_won_control_lost']['selected']] == [1]
    assert [r['seed'] for r in panel['swing']['control_won_candidate_lost']['selected']] == [4]
    stored = panel['swing']['candidate_won_control_lost']['selected'][0]
    assert stored['candidate'] == {'winner': 0, 'turns': 80, 'points': [10, 6, 6, 6], 'truncated': False}


def test_report_formatting_survives_all_win_and_all_loss_panels():
    for winner in (0, 1):
        rows = [game_row(seed, 0, winner, winner) for seed in range(3)]
        _, panels = attribution.analyse(iter(rows), 10, 1)
        text = attribution.describe('p', panels['p'])
        assert 'n/a' in text


def test_weighted_aggregate_matches_hand_computation():
    track = {'stolen_from': 0, 'stole': 0, 'discarded': 0, 'robber_on_my_tile': 0, 'trades_accepted_mine': 0,
             'trades_accepted_theirs': 0, 'offers_made': 0, 'bank_trades': 0, 'dev_bought': 0,
             'points_at_turn_40': 0, 'points_at_turn_60': 0, 'production': {}, 'first_city_turn': None,
             'final': {'longest_road': False, 'largest_army': False, 'my_points': 0}}
    a = {**track, 'discarded': 10, 'first_city_turn': 40}
    b = {**track, 'discarded': 4}
    out = replay.aggregate([a, b], [1, 3])
    assert out['discarded'] == pytest.approx((10 + 12) / 4)
    assert out['first_city_turn'] == 40
    assert out['no_city_share'] == pytest.approx(.75)


def test_registry_indexes_nested_readouts_and_legacy_power_fields(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'family-v1').mkdir(parents=True)
    (runs / 'family-v1' / 'readout.json').write_text(json.dumps({'pass': False, 'panels': [
        {'control': 'tactical', 'panel': 'mixed', 'paired_win_lift': .2, 'simultaneous_lcb': .19, 'pass': True},
        {'control': 'external', 'panel': 'mixed', 'paired_win_lift': None, 'pass': False}]}))
    (runs / 'screen-v1.json').write_text(json.dumps({'config': {'stage': 'dev'}, 'games': [], 'summary': {
        'pass': False, 'panels': {'a': {'paired_win_lift': .01, 'minimum_detectable_effect_80': .03, 'pass': False}}}}))
    out = tmp_path / 'out'
    import sys
    argv, sys.argv = sys.argv, ['x', '--runs', str(runs), '--out', str(out)]
    try:
        registry.main()
    finally:
        sys.argv = argv
    data = json.loads((out / 'registry.json').read_text())
    names = {r['name']: r for r in data['receipts']}
    readout = names['family-v1/readout']
    assert readout['pass'] is False
    assert readout['panels']['tactical/mixed']['paired_seed_lcb'] == .19
    assert readout['panels']['external/mixed']['pass'] is False
    assert names['screen-v1']['panels']['a']['zero_null_mde_80'] == .03
    assert 'zero-null mde80' in (out / 'LADDER.md').read_text()
