from catan_ai.inference import paired_normal_lcb, planned_seeds


def test_small_samples_cannot_claim_approximate_significance():
    assert paired_normal_lcb([1.] * 32) is None
    assert paired_normal_lcb([0.] * 128) < 0


def test_power_plan_accounts_for_variability_and_effect_size():
    assert planned_seeds([-.5, .5] * 16) > planned_seeds([0.] * 32)
    assert planned_seeds([-.5, .5] * 16, effect=.025) > planned_seeds([-.5, .5] * 16)


def test_variance_floor_does_not_certify_tiny_constant_gain():
    assert paired_normal_lcb([.01] * 128) < 0
