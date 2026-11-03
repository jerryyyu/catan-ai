import runpy
from pathlib import Path


def test_selfplay_intervals_resample_whole_games():
    sf = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/selfplay_features.py'))
    rows = [{'seed': g, 'seat': s, 'won': s == g % 4, 'pips': 10 + s, 'diversity': 4, 'ore_grain': 4,
             'brick_lumber': 4, 'port': False, 'pips_rank': 4 - s} for g in range(8) for s in range(4)]
    out = sf['clustered_intervals'](rows, resamples=50, seed=1)
    lo, hi, n = out['intervals_95']['by_seat/0/win_rate']
    assert 0 <= lo <= .25 <= hi <= 1 and n == 50
    assert 'mean_feature_winner_vs_loser/pips/2' in out['intervals_95']
