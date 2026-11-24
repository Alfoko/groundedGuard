from collections.abc import Sequence

import pytest

from groundguard.chunking import Chunk
from groundguard.llm import FakeLLMClient
from groundguard.verifiers import (
    LLMJudgeVerifier,
    NLIVerifier,
    SelfCheckVerifier,
    Verdict,
    _normalise_label,
)


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


def judge(reply: str) -> tuple[LLMJudgeVerifier, FakeLLMClient]:
    client = FakeLLMClient([reply])
    return LLMJudgeVerifier(client), client


def test_judge_supported_maps_passage_to_chunk() -> None:
    verifier, client = judge(
        '{"verdict": "supported", "passage": 2, "confidence": 0.8, "reason": "stated"}'
    )
    result = verifier.verify("Jobs retry failed Pods.", [A, B])
    assert result.verdict is Verdict.SUPPORTED
    assert result.score == pytest.approx(0.8)
    assert result.evidence_id == "b"
    assert result.rationale == "stated"
    prompt = client.calls[0][1]["content"]
    assert "[1] Pods share" in prompt
    assert "[2] Jobs retry" in prompt
    assert prompt.endswith("Claim: Jobs retry failed Pods.")


def test_judge_unsupported_score_is_inverted() -> None:
    verifier, _ = judge('{"verdict": "not_enough_info", "passage": null, "confidence": 0.9}')
    result = verifier.verify("claim", [A])
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert result.score == pytest.approx(0.1)
    assert result.evidence_id is None


def test_judge_accepts_fenced_json_and_ignores_bad_passage_numbers() -> None:
    verifier, _ = judge(
        '```json\n{"verdict": "contradicted", "passage": 7, "confidence": 1.0}\n```'
    )
    result = verifier.verify("claim", [A])
    assert result.verdict is Verdict.CONTRADICTED
    assert result.evidence_id is None


@pytest.mark.parametrize(
    "reply",
    [
        "I think it's supported.",
        '{"verdict": "maybe", "confidence": 0.5}',
        '{"verdict": "supported", "confidence": 1.5}',
    ],
)
def test_judge_fails_closed_on_bad_replies(reply: str) -> None:
    verifier, _ = judge(reply)
    result = verifier.verify("claim", [A])
    assert result.verdict is Verdict.NOT_ENOUGH_INFO
    assert result.score == 0.0


def test_judge_skips_llm_without_evidence() -> None:
    verifier, client = judge("unused")
    assert verifier.verify("claim", []).verdict is Verdict.NOT_ENOUGH_INFO
    assert client.calls == []


def test_self_check_flags_listed_sentences_in_one_call() -> None:
    client = FakeLLMClient(['{"unsupported": [2]}'], cost_per_call=0.03)
    results = SelfCheckVerifier(client).verify_batch(["Pods share a network.", "Jobs fly."], [A, B])
    assert [r.verdict for r in results] == [Verdict.SUPPORTED, Verdict.NOT_ENOUGH_INFO]
    assert sum(r.cost_usd or 0 for r in results) == pytest.approx(0.03)
    assert len(client.calls) == 1
    prompt = client.calls[0][1]["content"]
    assert "1. Pods share a network.\n2. Jobs fly." in prompt


def test_self_check_fails_closed() -> None:
    client = FakeLLMClient(["all good!"])
    results = SelfCheckVerifier(client).verify_batch(["a.", "b."], [A])
    assert all(r.verdict is Verdict.NOT_ENOUGH_INFO for r in results)


def test_nli_model_loads_once_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import groundguard.verifiers as verifiers

    loads = []
    lock = threading.Lock()

    def slow_loader(name: str) -> TableScorer:
        with lock:
            loads.append(name)
        time.sleep(0.05)
        return TableScorer({A.text: (0.9, 0.05, 0.05)})

    monkeypatch.setattr(verifiers, "_hf_scorer", slow_loader)
    verifier = NLIVerifier()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: verifier.verify("claim", [A]), range(16)))
    assert len(loads) == 1
    assert all(r.supported for r in results)
