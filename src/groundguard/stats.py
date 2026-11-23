"""Confidence intervals and significance tests for the eval metrics.

Every rate we report is a ratio of sums over answers: hallucinated shown
claims / shown claims, declined answerable / answerable, and so on. So each
system's result is kept as two per-answer arrays, a numerator and a
denominator, and the statistic is sum(num) / sum(den). Resampling answers
rather than claims matters: claims within one answer are correlated, and
treating them as independent would give intervals that are too narrow.

bootstrap_ci            percentile bootstrap over answers
paired_permutation_test two systems on the same questions; under the null
                        the system labels are exchangeable within each
                        question, so we randomly swap them per question
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _as_arrays(num: npt.ArrayLike, den: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    if n.shape != d.shape or n.ndim != 1:
        raise ValueError("num and den must be 1-D arrays of the same length")
    return n, d


def ratio(num: npt.ArrayLike, den: npt.ArrayLike) -> float:
    n, d = _as_arrays(num, den)
    total = d.sum()
    return float(n.sum() / total) if total else 0.0


def _ratios(n: FloatArray, d: FloatArray) -> FloatArray:
    """Row-wise sum(n)/sum(d) for 2-D arrays, 0 where the denominator is 0."""
    num, den = n.sum(axis=-1), d.sum(axis=-1)
    out: FloatArray = np.zeros(num.shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    num: npt.ArrayLike,
    den: npt.ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    n, d = _as_arrays(num, den)
    value = ratio(n, d)
    if len(n) == 0:
        return Estimate(value, value, value, 0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    # Chunked so a 10k x 3k index matrix doesn't have to sit in memory at once.
    for start in range(0, n_resamples, 1000):
        stop = min(start + 1000, n_resamples)
        idx = rng.integers(0, len(n), size=(stop - start, len(n)))
        stats[start:stop] = _ratios(n[idx], d[idx])
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return Estimate(value, float(low), float(high), len(n))


@dataclass(frozen=True)
class Comparison:
    a: float
    b: float
    diff: float
    """b - a"""
    p_value: float
    n: int


def paired_permutation_test(
    num_a: npt.ArrayLike,
    den_a: npt.ArrayLike,
    num_b: npt.ArrayLike,
    den_b: npt.ArrayLike,
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Two-sided test of whether systems A and B differ on a ratio metric.

    Inputs are aligned by question: index i in every array is the same
    question answered by A and by B.
    """
    na, da = _as_arrays(num_a, den_a)
    nb, db = _as_arrays(num_b, den_b)
    if len(na) != len(nb):
        raise ValueError("systems must be evaluated on the same questions")
    a, b = ratio(na, da), ratio(nb, db)
    observed = b - a
    if len(na) == 0:
        return Comparison(a, b, observed, 1.0, 0)

    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_permutations, 1000):
        size = min(1000, n_permutations - start)
        swap = rng.random((size, len(na))) < 0.5
        pa_n = np.where(swap, nb, na)
        pa_d = np.where(swap, db, da)
        pb_n = np.where(swap, na, nb)
        pb_d = np.where(swap, da, db)
        diffs = _ratios(pb_n, pb_d) - _ratios(pa_n, pa_d)
        # Small tolerance so ties with the observed value count as extreme.
        extreme += int(np.sum(np.abs(diffs) >= abs(observed) - 1e-12))
    # +1 in numerator and denominator: the observed labelling is itself one
    # of the permutations, and it keeps p from ever being exactly zero.
    p_value = (extreme + 1) / (n_permutations + 1)
    return Comparison(a, b, observed, p_value, len(na))
