import numpy as np
import pytest

from groundguard.stats import bootstrap_ci, paired_permutation_test, ratio


def test_ratio_of_sums() -> None:
    assert ratio([1, 0, 2], [2, 0, 6]) == pytest.approx(3 / 8)
    assert ratio([], []) == 0.0
    with pytest.raises(ValueError):
        ratio([1, 2], [1])


def test_bootstrap_interval_contains_estimate_and_narrows_with_n() -> None:
    rng = np.random.default_rng(1)
    small = rng.random(50) < 0.3
    large = rng.random(5000) < 0.3
    est_small = bootstrap_ci(small, np.ones(50), n_resamples=2000)
    est_large = bootstrap_ci(large, np.ones(5000), n_resamples=2000)
    assert est_small.low <= est_small.value <= est_small.high
    assert (est_large.high - est_large.low) < (est_small.high - est_small.low) / 5


def test_bootstrap_width_matches_normal_approximation() -> None:
    # For a proportion p with n samples the 95% interval is about
    # +/- 1.96 * sqrt(p(1-p)/n); with p=0.3, n=1000 that's +/- 0.0284.
    rng = np.random.default_rng(2)
    x = (rng.random(1000) < 0.3).astype(float)
    est = bootstrap_ci(x, np.ones(1000), n_resamples=5000)
    assert (est.high - est.low) / 2 == pytest.approx(0.0284, rel=0.15)


def test_bootstrap_coverage_is_close_to_nominal() -> None:
    rng = np.random.default_rng(3)
    trials, hits = 200, 0
    for i in range(trials):
        x = (rng.random(200) < 0.3).astype(float)
        est = bootstrap_ci(x, np.ones(200), n_resamples=1000, seed=i)
        hits += est.low <= 0.3 <= est.high
    assert 0.90 <= hits / trials <= 0.99


def test_bootstrap_resamples_whole_answers_for_ratio_metrics() -> None:
    # Two answers: one with 9 shown claims all fine, one with 1 claim that's
    # bad. Resampling answers can produce rates of 0, 0.1 or 1, never e.g. 0.5.
    est = bootstrap_ci([0, 1], [9, 1], n_resamples=2000)
    assert est.value == pytest.approx(0.1)
    assert est.low == 0.0
    assert est.high == 1.0


def test_bootstrap_is_reproducible_with_seed() -> None:
    x = np.arange(30) % 4 == 0
    assert bootstrap_ci(x, np.ones(30), seed=7) == bootstrap_ci(x, np.ones(30), seed=7)


def test_permutation_test_detects_a_real_difference() -> None:
    rng = np.random.default_rng(4)
    a = (rng.random(2000) < 0.30).astype(float)
    b = (rng.random(2000) < 0.20).astype(float)
    ones = np.ones(2000)
    result = paired_permutation_test(a, ones, b, ones, n_permutations=2000)
    assert result.diff == pytest.approx(b.mean() - a.mean())
    assert result.p_value < 0.001


def test_small_n_looks_different_but_cannot_be_told_apart() -> None:
    # 20 questions: system A hallucinates on 6, system B on 3. A 15-point gap
    # looks like a big win, but with n=20 it's well within noise.
    a = np.array([1] * 6 + [0] * 14, dtype=float)
    b = np.array([1] * 3 + [0] * 17, dtype=float)
    ones = np.ones(20)
    result = paired_permutation_test(a, ones, b, ones, n_permutations=5000)
    assert result.diff == pytest.approx(-0.15)
    assert result.p_value > 0.05
    ci_a = bootstrap_ci(a, ones, n_resamples=5000)
    ci_b = bootstrap_ci(b, ones, n_resamples=5000)
    assert ci_a.low < ci_b.high  # the intervals overlap

    # The same rates on 100x the questions are clearly different.
    big = paired_permutation_test(
        np.tile(a, 100), np.ones(2000), np.tile(b, 100), np.ones(2000), n_permutations=2000
    )
    assert big.p_value < 0.001


def test_permutation_test_false_positive_rate_under_the_null() -> None:
    rng = np.random.default_rng(5)
    rejections = 0
    trials = 200
    for i in range(trials):
        a = (rng.random(100) < 0.3).astype(float)
        b = (rng.random(100) < 0.3).astype(float)
        ones = np.ones(100)
        p = paired_permutation_test(a, ones, b, ones, n_permutations=500, seed=i).p_value
        rejections += p < 0.05
    assert rejections / trials <= 0.10


def test_identical_systems_are_not_different() -> None:
    x = np.array([1, 0, 0, 1, 0], dtype=float)
    result = paired_permutation_test(x, np.ones(5), x, np.ones(5), n_permutations=500)
    assert result.diff == 0.0
    assert result.p_value == 1.0


def test_permutation_test_requires_paired_inputs() -> None:
    with pytest.raises(ValueError):
        paired_permutation_test([1], [1], [1, 0], [1, 1])
