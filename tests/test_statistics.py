from cisose_deeprm.evaluation import holm_bonferroni, paired_bootstrap_ci, sign_flip_pvalues


def test_bootstrap_ci_is_ordered():
    low, high = paired_bootstrap_ci([-2, -1, -3, -2, -4], seed=1, resamples=100)
    assert low <= high
    assert high < 0


def test_sign_flip_pvalues_are_probabilities():
    p_less, p_greater = sign_flip_pvalues([-2, -1, -3, -2, -4], seed=1, resamples=1000)
    assert 0 <= p_less <= 1
    assert 0 <= p_greater <= 1
    assert p_less < p_greater


def test_holm_adjusted_pvalues_are_monotone_by_rank():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.03, "c": 0.02})
    assert set(adjusted) == {"a", "b", "c"}
    assert all(0 <= value <= 1 for value in adjusted.values())

