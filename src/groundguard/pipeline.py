"""Retrieve, generate, then guard: the full question-answering path."""

from __future__ import annotations

from dataclasses import dataclass

from groundguard.generation import Answer, AnswerGenerator
from groundguard.guard import Decision, Guard, GuardResult
from groundguard.retrieval import HybridRetriever, ScoredChunk


@dataclass(frozen=True)
class GuardedResponse:
    question: str
    hits: list[ScoredChunk]
    answer: Answer
    guard: GuardResult | None
    """None when the pipeline runs without a guard (the baseline)."""

    @property
    def text(self) -> str:
        if self.guard is None:
            return self.answer.text
        return self.guard.message

    @property
    def declined(self) -> bool:
        if self.guard is None:
            return not self.answer.claims
        return self.guard.decision is Decision.DECLINE


class GuardedAssistant:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        guard: Guard | None,
        k: int = 6,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.guard = guard
        self.k = k

    def ask(self, question: str) -> GuardedResponse:
        hits = self.retriever.search(question, k=self.k)
        answer = self.generator.generate(question, [h.chunk for h in hits])
        result = self.guard.check(answer) if self.guard is not None else None
        return GuardedResponse(question, hits, answer, result)
