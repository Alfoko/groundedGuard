import json

import pytest

from groundguard.chunking import Chunk
from groundguard.generation import (
    AnswerGenerator,
    GenerationError,
    build_messages,
    parse_answer,
)
from groundguard.llm import FakeLLMClient


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source=f"{cid}.md", section=cid.title(), start=0, end=len(text), text=text)


SOURCES = [
    chunk("pods", "A Pod is the smallest deployable unit in Kubernetes."),
    chunk("jobs", "A Job runs Pods until a given number of them succeed."),
]

GOOD_REPLY = json.dumps(
    {
        "answerable": True,
        "claims": [
            {"text": "A Pod is the smallest deployable unit.", "sources": [1]},
            {"text": "A Job runs Pods until enough succeed.", "sources": [2, 1]},
        ],
    }
)


def test_prompt_numbers_sources_in_order() -> None:
    messages = build_messages("What is a Pod?", SOURCES)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "[1] Pods (pods.md)\nA Pod is" in user
    assert "[2] Jobs (jobs.md)" in user
    assert user.index("[1]") < user.index("[2]")
    assert user.endswith("Question: What is a Pod?")


def test_parse_accepts_plain_and_fenced_json() -> None:
    plain = parse_answer(GOOD_REPLY)
    fenced = parse_answer(f"```json\n{GOOD_REPLY}\n```")
    assert plain == fenced
    assert plain.claims[1].sources == [2, 1]


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is yes.",
        '{"claims": []}',  # answerable missing
        '{"answerable": true, "claims": [{"text": "", "sources": [1]}]}',
        '{"answerable": true, "claims": [{"text": "x", "sources": "1"}]}',
    ],
)
def test_parse_rejects_bad_replies(reply: str) -> None:
    with pytest.raises(GenerationError):
        parse_answer(reply)


def test_generate_maps_source_numbers_to_chunks() -> None:
    fake = FakeLLMClient([GOOD_REPLY], cost_per_call=0.001)
    answer = AnswerGenerator(fake).generate("What do Pods and Jobs do?", SOURCES)

    assert answer.structured.answerable
    assert answer.claims[0].text == "A Pod is the smallest deployable unit."
    assert answer.chunk_for(2) == SOURCES[1]
    assert answer.chunk_for(3) is None
    assert answer.chunk_for(0) is None
    assert answer.text.startswith("A Pod is")
    assert answer.usage.cost_usd == pytest.approx(0.001)


def test_generate_retries_once_with_feedback() -> None:
    fake = FakeLLMClient(["Sure! Pods are small.", GOOD_REPLY], cost_per_call=0.001)
    answer = AnswerGenerator(fake).generate("What is a Pod?", SOURCES)

    assert len(answer.claims) == 2
    assert len(answer.completions) == 2
    assert answer.usage.cost_usd == pytest.approx(0.002)
    retry_messages = fake.calls[1]
    assert retry_messages[-2] == {"role": "assistant", "content": "Sure! Pods are small."}
    assert "not valid JSON" in retry_messages[-1]["content"]


def test_generate_gives_up_after_max_attempts() -> None:
    fake = FakeLLMClient(["nope", "still nope"])
    with pytest.raises(GenerationError, match="2 attempts"):
        AnswerGenerator(fake).generate("What is a Pod?", SOURCES)


def test_unanswerable_reply_has_no_claims() -> None:
    fake = FakeLLMClient(['{"answerable": false, "claims": []}'])
    answer = AnswerGenerator(fake).generate("Who wrote Kubernetes?", SOURCES)
    assert not answer.structured.answerable
    assert answer.claims == []
    assert answer.text == ""
