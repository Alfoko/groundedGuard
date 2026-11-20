from collections.abc import Sequence

import pytest

from groundguard.chunking import Chunk
from groundguard.verifiers import NLIVerifier, Verdict, _normalise_label


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source="d.md", section="D", start=0, end=len(text), text=text)


class TableScorer:
    """Fake NLI model: looks up (entail, neutral, contradict) by premise text."""

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self.table = table
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        self.calls.append(list(pairs))
        rows = []
        for premise, _ in pairs:
            e, n, c = self.table[premise]
            rows.append({"entailment": e, "neutral": n, "contradiction": c})
        return rows


A = chunk("a", "Pods share a network namespace.")
B = chunk("b", "Jobs retry failed Pods.")


def test_best_supporting_chunk_wins() -> None:
    scorer = TableScorer({A.text: (0.2, 0.7, 0.1), B.text: (0.9, 0.05, 0.05)})
    result = NLIVerifier(scorer=scorer).verify("Jobs retry Pods.", [A, B])
    assert result.verdict is Verdict.SUPPORTED
    assert result.score == pytest.approx(0.9)
    assert result.evidence_id == "b"
    assert scorer.calls == [[(A.text, "Jobs retry Pods."), (B.text, "Jobs retry Pods.")]]


def test_below_threshold_is_not_enough_info() -> None:
    scorer = TableScorer({A.text: (0.4, 0.5, 0.1)})
    result = NLIVerifier(scorer=scorer, threshold=0.5).verify("claim", [A])
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert not result.supported


def test_threshold_is_configurable() -> None:
    scorer = TableScorer({A.text: (0.4, 0.5, 0.1)})
    assert NLIVerifier(scorer=scorer, threshold=0.3).verify("claim", [A]).supported


def test_strong_contradiction_is_reported() -> None:
    scorer = TableScorer({A.text: (0.05, 0.15, 0.8), B.text: (0.1, 0.8, 0.1)})
    result = NLIVerifier(scorer=scorer).verify("Pods have separate networks.", [A, B])
    assert result.verdict is Verdict.CONTRADICTED
    assert result.evidence_id == "a"


def test_weak_contradiction_stays_neutral() -> None:
    scorer = TableScorer({A.text: (0.2, 0.45, 0.35)})
    assert NLIVerifier(scorer=scorer).verify("claim", [A]).verdict is Verdict.NOT_ENOUGH_INFO


def test_no_evidence_skips_the_model() -> None:
    scorer = TableScorer({})
    result = NLIVerifier(scorer=scorer).verify("claim", [])
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ENTAILMENT", "entailment"), ("neutral", "neutral"), ("Contradiction", "contradiction")],
)
def test_label_names_are_normalised(raw: str, expected: str) -> None:
    assert _normalise_label(raw) == expected


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        _normalise_label("LABEL_0")
