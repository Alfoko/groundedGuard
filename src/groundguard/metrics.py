"""Evaluation metrics, computed from per-answer records.

Two kinds of record feed the metrics:

ClaimRecord   one claim with its gold label and the verifier's call; used for
              verifier precision/recall, where the positive class is
              "hallucinated" (i.e. we measure how well the verifier catches
              bad claims).
AnswerRecord  what one system returned for one question; used for the
              user-facing rates, latency and cost.

The hallucinated-claim rate is measured over claims the user actually sees:
claims the guard dropped don't count against it, and a declined answer shows
no claims at all. That's why over-refusal has to be reported next to it;
declining everything would score a perfect 0%.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClaimRecord:
    hallucinated: bool
    """Gold label."""
    flagged: bool
    """The verifier said the claim is not supported."""


@dataclass(frozen=True)
class AnswerRecord:
    answerable: bool
    """Gold: the question can be answered from the sources."""
    declined: bool
    shown_claims: int
    shown_hallucinated: int
    latency_s: float
    cost_usd: float | None = None


@dataclass(frozen=True)
class VerifierScores:
    precision: float
    recall: float
    f1: float
    support: int
    """Number of gold-hallucinated claims."""


def verifier_scores(records: Sequence[ClaimRecord]) -> VerifierScores:
    tp = sum(r.hallucinated and r.flagged for r in records)
    fp = sum(r.flagged and not r.hallucinated for r in records)
    fn = sum(r.hallucinated and not r.flagged for r in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return VerifierScores(precision, recall, f1, tp + fn)


def hallucinated_claim_rate(records: Sequence[AnswerRecord]) -> float:
    shown = sum(r.shown_claims for r in records)
    return sum(r.shown_hallucinated for r in records) / shown if shown else 0.0


def answer_hallucination_rate(records: Sequence[AnswerRecord]) -> float:
    """Share of non-declined answers that show at least one hallucinated claim."""
    answered = [r for r in records if not r.declined]
    if not answered:
        return 0.0
    return sum(r.shown_hallucinated > 0 for r in answered) / len(answered)


def over_refusal_rate(records: Sequence[AnswerRecord]) -> float:
    """Share of answerable questions the system declined."""
    answerable = [r for r in records if r.answerable]
    if not answerable:
        return 0.0
    return sum(r.declined for r in answerable) / len(answerable)


def latency_percentiles(records: Sequence[AnswerRecord]) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    p50, p95 = np.percentile([r.latency_s for r in records], [50, 95])
    return float(p50), float(p95)


def mean_cost(records: Sequence[AnswerRecord]) -> float | None:
    """Mean cost per answer, or None if any answer's cost is unknown."""
    costs = [r.cost_usd for r in records]
    if not costs or any(c is None for c in costs):
        return None
    return sum(c for c in costs if c is not None) / len(costs)


@dataclass(frozen=True)
class SystemMetrics:
    n: int
    hallucinated_claim_rate: float
    answer_hallucination_rate: float
    over_refusal_rate: float
    latency_p50_s: float
    latency_p95_s: float
    cost_per_answer_usd: float | None


def summarise(records: Sequence[AnswerRecord]) -> SystemMetrics:
    p50, p95 = latency_percentiles(records)
    return SystemMetrics(
        n=len(records),
        hallucinated_claim_rate=hallucinated_claim_rate(records),
        answer_hallucination_rate=answer_hallucination_rate(records),
        over_refusal_rate=over_refusal_rate(records),
        latency_p50_s=p50,
        latency_p95_s=p95,
        cost_per_answer_usd=mean_cost(records),
    )
