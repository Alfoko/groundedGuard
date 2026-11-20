"""Answer generation over numbered sources, with structured JSON output.

The model only ever sees the retrieved chunks, numbered [1]..[k], and has to
return one JSON object listing its claims and the source numbers behind each.
Asking for claims up front means the guard doesn't have to split free text
into claims afterwards, which is where a lot of noise would otherwise come in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from groundguard.chunking import Chunk
from groundguard.llm import Completion, LLMClient, Message, Usage


class Claim(BaseModel):
    text: str = Field(min_length=1)
    sources: list[int] = Field(default_factory=list)
    """1-based source numbers as shown in the prompt. Not validated here: an
    out-of-range number is a citation error for the guard to catch."""


class StructuredAnswer(BaseModel):
    answerable: bool
    claims: list[Claim] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You answer questions about software documentation using only the numbered \
sources you are given. Do not use outside knowledge.

Reply with a single JSON object and nothing else:
{"answerable": true, "claims": [{"text": "...", "sources": [1]}]}

Rules:
- Each claim is one self-contained factual statement, written so it makes \
sense on its own.
- "sources" lists the numbers of every source that supports the claim. \
Every claim needs at least one.
- Together, the claims should answer the question directly.
- If the sources do not contain the answer, reply with \
{"answerable": false, "claims": []}."""


def format_sources(chunks: Sequence[Chunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{number}] {chunk.section} ({chunk.source})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[Message]:
    user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class GenerationError(RuntimeError):
    pass


def parse_answer(text: str) -> StructuredAnswer:
    """Parse the model's reply, tolerating a Markdown code fence around it."""
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"reply is not valid JSON: {exc}") from exc
    try:
        return StructuredAnswer.model_validate(data)
    except ValidationError as exc:
        raise GenerationError(f"reply does not match the schema: {exc}") from exc


@dataclass(frozen=True)
class Answer:
    question: str
    sources: list[Chunk]
    structured: StructuredAnswer
    completions: list[Completion]

    @property
    def claims(self) -> list[Claim]:
        return self.structured.claims

    @property
    def text(self) -> str:
        return " ".join(c.text for c in self.claims)

    def chunk_for(self, number: int) -> Chunk | None:
        """Look up a 1-based source number; None if the model made it up."""
        if 1 <= number <= len(self.sources):
            return self.sources[number - 1]
        return None

    @property
    def usage(self) -> Usage:
        return Usage(
            prompt_tokens=sum(c.usage.prompt_tokens for c in self.completions),
            completion_tokens=sum(c.usage.completion_tokens for c in self.completions),
            latency_s=sum(c.usage.latency_s for c in self.completions),
            cost_usd=_sum_costs(self.completions),
        )


def _sum_costs(completions: Sequence[Completion]) -> float | None:
    costs = [c.usage.cost_usd for c in completions]
    if any(c is None for c in costs):
        return None
    return sum(c for c in costs if c is not None)


class AnswerGenerator:
    def __init__(self, client: LLMClient, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.client = client
        self.max_attempts = max_attempts

    def generate(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        messages = build_messages(question, chunks)
        completions: list[Completion] = []
        error: GenerationError | None = None
        for _ in range(self.max_attempts):
            completion = self.client.complete(messages, json_mode=True)
            completions.append(completion)
            try:
                structured = parse_answer(completion.text)
            except GenerationError as exc:
                error = exc
                # Show the model its own reply and what was wrong with it;
                # this fixes most schema slips in one more round.
                messages = [
                    *messages,
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": f"{exc}. Reply again with valid JSON only."},
                ]
                continue
            return Answer(question, list(chunks), structured, completions)
        raise GenerationError(
            f"no valid answer after {self.max_attempts} attempts: {error}"
        ) from error
