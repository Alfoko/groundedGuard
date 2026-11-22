import pytest

from groundguard.metrics import (
    AnswerRecord,
    ClaimRecord,
    answer_hallucination_rate,
    hallucinated_claim_rate,
    latency_percentiles,
    mean_cost,
    over_refusal_rate,
    summarise,
    verifier_scores,
)


def ans(
    shown: int = 2,
    bad: int = 0,
    declined: bool = False,
    answerable: bool = True,
    latency: float = 1.0,
    cost: float | None = 0.01,
) -> AnswerRecord:
    return AnswerRecord(answerable, declined, shown, bad, latency, cost)


def test_verifier_scores_treat_hallucinated_as_positive() -> None:
    records = (
        [ClaimRecord(True, True)] * 3  # caught
        + [ClaimRecord(True, False)]  # missed
        + [ClaimRecord(False, True)] * 2  # false alarm
        + [ClaimRecord(False, False)] * 10
    )
    scores = verifier_scores(records)
    assert scores.precision == pytest.approx(3 / 5)
    assert scores.recall == pytest.approx(3 / 4)
    assert scores.f1 == pytest.approx(2 * 0.6 * 0.75 / 1.35)
    assert scores.support == 4


def test_verifier_scores_with_nothing_flagged() -> None:
    scores = verifier_scores([ClaimRecord(True, False), ClaimRecord(False, False)])
    assert (scores.precision, scores.recall, scores.f1) == (0.0, 0.0, 0.0)


def test_claim_rate_counts_only_shown_claims() -> None:
    records = [ans(shown=4, bad=1), ans(shown=0, declined=True), ans(shown=6, bad=0)]
    assert hallucinated_claim_rate(records) == pytest.approx(1 / 10)


def test_answer_rate_ignores_declined_answers() -> None:
    records = [ans(bad=1), ans(bad=0), ans(shown=0, declined=True)]
    assert answer_hallucination_rate(records) == pytest.approx(1 / 2)


def test_over_refusal_only_counts_answerable_questions() -> None:
    records = [
        ans(declined=True),
        ans(declined=False),
        ans(declined=True, answerable=False),  # a correct refusal
        ans(declined=False),
    ]
    assert over_refusal_rate(records) == pytest.approx(1 / 3)


def test_declining_everything_is_visible_in_over_refusal() -> None:
    records = [ans(shown=0, declined=True) for _ in range(5)]
    assert hallucinated_claim_rate(records) == 0.0
    assert over_refusal_rate(records) == 1.0


def test_latency_percentiles() -> None:
    records = [ans(latency=float(x)) for x in range(1, 101)]
    p50, p95 = latency_percentiles(records)
    assert p50 == pytest.approx(50.5)
    assert p95 == pytest.approx(95.05)


def test_mean_cost_is_unknown_if_any_cost_is() -> None:
    assert mean_cost([ans(cost=0.02), ans(cost=0.04)]) == pytest.approx(0.03)
    assert mean_cost([ans(cost=0.02), ans(cost=None)]) is None
    assert mean_cost([]) is None


def test_empty_inputs_do_not_divide_by_zero() -> None:
    metrics = summarise([])
    assert metrics.n == 0
    assert metrics.hallucinated_claim_rate == 0.0
    assert metrics.over_refusal_rate == 0.0
    assert metrics.cost_per_answer_usd is None
