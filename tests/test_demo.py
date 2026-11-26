import json
from collections.abc import Sequence

import pytest

from groundguard.chunking import Chunk
from groundguard.demo import (
    CLAIM_HEADERS,
    EXAMPLE_ANSWER,
    EXAMPLE_SOURCES,
    ask,
    check_answer,
    parse_sources,
)
from groundguard.llm import FakeLLMClient
from groundguard.retrieval import BM25Index, HybridRetriever
from groundguard.verifiers import Verdict, Verification


class WordVerifier:
    """Supported iff every word of the claim appears in one evidence chunk."""

    name = "words"

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        words = [w.strip(".,").lower() for w in claim.split()]
        for chunk in evidence:
            if all(w in chunk.text.lower() for w in words):
                return Verification(Verdict.SUPPORTED, 1.0, chunk.id)
        return Verification(Verdict.NOT_ENOUGH_INFO, 0.0)


SOURCES = "A Pod is the smallest deployable unit.\n\n\nA Job retries failed Pods."


def test_sources_split_on_blank_lines() -> None:
    chunks = parse_sources(SOURCES)
    assert [c.text for c in chunks] == [
        "A Pod is the smallest deployable unit.",
        "A Job retries failed Pods.",
    ]
    assert chunks[1].section == "Source 2"


def test_guard_on_drops_the_unsupported_claim() -> None:
    view = check_answer(
        WordVerifier(), SOURCES, "A Pod is the smallest deployable unit. Pods run on Mars.", True
    )
    assert view.message.startswith("**Decision: partial** · 50% of claims supported")
    assert "Mars" not in view.message.split("\n\n", 1)[1]
    assert [row[3] for row in view.claims] == ["supported", "not_enough_info"]
    assert [row[5] for row in view.claims] == ["yes", "no"]
    assert all(len(row) == len(CLAIM_HEADERS) for row in view.claims)
    assert "**[2] Source 2**" in view.sources


def test_guard_off_shows_everything_unchecked() -> None:
    view = check_answer(WordVerifier(), SOURCES, "Pods run on Mars.", False)
    assert "Pods run on Mars. [1][2]" in view.message
    assert view.claims[0][3:] == ["not checked", "", "yes"]


def test_strict_policy_declines() -> None:
    answer = "A Pod is the smallest deployable unit. Pods run on Mars."
    view = check_answer(WordVerifier(), SOURCES, answer, True, mode="strict")
    assert view.message.startswith("**Decision: decline**")


def test_empty_input_gives_a_hint() -> None:
    assert check_answer(WordVerifier(), "", "x.", True).claims == []
    assert "Paste" in check_answer(WordVerifier(), SOURCES, "  ", True).message


def test_ask_uses_retrieved_chunks_and_citations() -> None:
    chunks = [
        Chunk(c.id, c.source, c.section, 0, len(c.text), c.text) for c in parse_sources(SOURCES)
    ]
    reply = json.dumps(
        {
            "answerable": True,
            "claims": [
                {"text": "A Job retries failed Pods.", "sources": [1]},
                {"text": "A Job retries forever.", "sources": [7]},
            ],
        }
    )
    retriever = HybridRetriever(BM25Index(chunks))
    view = ask(WordVerifier(), retriever, FakeLLMClient([reply]), "does a job retry", True, k=1)
    assert [row[2] for row in view.claims] == ["[1]", "[7]"]
    assert view.claims[1][3] != "supported"
    assert "Source 2" in view.sources


def test_gradio_app_builds_and_runs_the_check_tab() -> None:
    pytest.importorskip("gradio")
    from groundguard.demo import build_demo

    demo = build_demo(WordVerifier())
    fn = next(f for f in demo.fns.values() if len(f.inputs) == 5)
    message, rows, _ = fn.fn(EXAMPLE_SOURCES, EXAMPLE_ANSWER, True, "drop", 0.5)
    assert message.startswith("**Decision:")
    assert len(rows) == 3
