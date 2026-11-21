from collections.abc import Sequence

import pytest

from groundguard.chunking import Chunk
from groundguard.generation import Answer, Claim, StructuredAnswer
from groundguard.guard import (
    DECLINE_TEXT,
    Decision,
    Guard,
    GuardConfig,
    GuardResult,
    PolicyMode,
)
from groundguard.verifiers import Verdict, Verification


class ScriptedVerifier:
    """Returns a fixed verdict per claim text; records what it was asked."""

    name = "scripted"

    def __init__(self, verdicts: dict[str, Verdict]) -> None:
        self.verdicts = verdicts
        self.calls: list[str] = []

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        self.calls.append(claim)
        verdict = self.verdicts[claim]
        return Verification(verdict, 1.0 if verdict is Verdict.SUPPORTED else 0.0)


SOURCES = [
    Chunk(id=f"c{i}", source="d.md", section="D", start=0, end=1, text="t") for i in range(3)
]


def answer(*claims: tuple[str, list[int]], answerable: bool = True) -> Answer:
    structured = StructuredAnswer(
        answerable=answerable, claims=[Claim(text=t, sources=s) for t, s in claims]
    )
    return Answer("q", SOURCES, structured, [])


OK = Verdict.SUPPORTED
NEI = Verdict.NOT_ENOUGH_INFO
BAD = Verdict.CONTRADICTED


def run(
    ans: Answer,
    verdicts: dict[str, Verdict],
    mode: PolicyMode = PolicyMode.DROP,
    min_support: float = 0.5,
) -> tuple[GuardResult, ScriptedVerifier]:
    verifier = ScriptedVerifier(verdicts)
    config = GuardConfig(mode=mode, min_support=min_support)
    return Guard(verifier, config).check(ans), verifier


def test_all_supported_answers_in_full() -> None:
    result, _ = run(answer(("A.", [1]), ("B.", [2])), {"A.": OK, "B.": OK})
    assert result.decision is Decision.ANSWER
    assert result.message == "A. B."
    assert result.support_ratio == 1.0
    assert result.dropped == []


def test_drop_mode_removes_unsupported_claims() -> None:
    result, _ = run(answer(("A.", [1]), ("B.", [2]), ("C.", [3])), {"A.": OK, "B.": NEI, "C.": OK})
    assert result.decision is Decision.PARTIAL
    assert result.message == "A. C."
    assert [r.claim.text for r in result.dropped] == ["B."]


def test_strict_mode_rejects_on_any_unsupported_claim() -> None:
    result, _ = run(
        answer(("A.", [1]), ("B.", [2]), ("C.", [3])),
        {"A.": OK, "B.": BAD, "C.": OK},
        mode=PolicyMode.STRICT,
    )
    assert result.decision is Decision.DECLINE
    assert result.kept == []
    assert "- B. (contradicted by the sources)" in result.message


def test_declines_below_min_support() -> None:
    result, _ = run(
        answer(("A.", [1]), ("B.", [2]), ("C.", [3])),
        {"A.": OK, "B.": NEI, "C.": NEI},
        min_support=0.5,
    )
    assert result.decision is Decision.DECLINE
    assert result.message.startswith(DECLINE_TEXT)


def test_min_support_boundary_is_inclusive() -> None:
    result, _ = run(answer(("A.", [1]), ("B.", [2])), {"A.": OK, "B.": NEI}, min_support=0.5)
    assert result.decision is Decision.PARTIAL


def test_declines_when_nothing_is_supported_even_with_zero_threshold() -> None:
    result, _ = run(answer(("A.", [1])), {"A.": NEI}, min_support=0.0)
    assert result.decision is Decision.DECLINE


def test_bad_citations_never_reach_the_verifier() -> None:
    result, verifier = run(
        answer(("A.", [1]), ("No source.", []), ("Fake source.", [9])),
        {"A.": OK},
        min_support=0.0,
    )
    assert verifier.calls == ["A."]
    assert result.decision is Decision.PARTIAL
    reasons = {r.claim.text: r.verification.verdict for r in result.dropped}
    assert reasons == {"No source.": NEI, "Fake source.": NEI}


def test_decline_message_explains_each_problem() -> None:
    result, _ = run(
        answer(("No source.", []), ("Fake source.", [9]), ("Wrong.", [1]), ("Vague.", [2])),
        {"Wrong.": BAD, "Vague.": NEI},
    )
    assert result.decision is Decision.DECLINE
    assert result.message.splitlines() == [
        DECLINE_TEXT,
        "These points couldn't be confirmed:",
        "- No source. (no source cited)",
        "- Fake source. (cited a source that wasn't provided)",
        "- Wrong. (contradicted by the sources)",
        "- and 1 more",
    ]


def test_unanswerable_answer_declines_without_verifying() -> None:
    result, verifier = run(answer(answerable=False), {})
    assert result.decision is Decision.DECLINE
    assert verifier.calls == []
    assert "don't cover this question" in result.message


def test_config_validates_threshold() -> None:
    with pytest.raises(ValueError):
        GuardConfig(min_support=1.5)
