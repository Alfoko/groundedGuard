"""HTTP service: /ask, /verify and /health (install the `service` extra).

/ask runs the full pipeline over the indexed corpus. /verify is for callers
with their own RAG stack: they send an answer plus the sources it was based
on and get claim-by-claim verdicts back. Both take an optional policy so a
caller can switch between drop and strict, or change the support threshold,
per request.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from groundguard.chunking import Chunk
from groundguard.generation import Answer, AnswerGenerator, Claim, StructuredAnswer
from groundguard.guard import Guard, GuardConfig, GuardResult
from groundguard.llm import LLMClient
from groundguard.ragtruth import sentence_spans
from groundguard.retrieval import HybridRetriever
from groundguard.verifiers import Verifier


class SourceIn(BaseModel):
    text: str = Field(min_length=1)
    id: str | None = None
    title: str = ""


class ClaimIn(BaseModel):
    text: str = Field(min_length=1)
    sources: list[int] = Field(default_factory=list)
    """1-based indexes into the request's `sources` list."""


class VerifyRequest(BaseModel):
    sources: list[SourceIn] = Field(min_length=1)
    answer: str | None = None
    """Free text; split into sentences, each checked against all sources."""
    claims: list[ClaimIn] | None = None
    """Pre-split claims with their own citations; takes precedence over `answer`."""
    policy: GuardConfig = Field(default_factory=GuardConfig)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=20)
    policy: GuardConfig = Field(default_factory=GuardConfig)


class ClaimOut(BaseModel):
    text: str
    cited: list[int]
    verdict: str
    score: float
    kept: bool
    evidence_id: str | None
    rationale: str


class SourceOut(BaseModel):
    number: int
    id: str
    source: str
    section: str
    text: str


class GuardOut(BaseModel):
    decision: str
    message: str
    support_ratio: float
    claims: list[ClaimOut]
    sources: list[SourceOut]
    latency_ms: float
    cost_usd: float | None


def _guard_out(
    result: GuardResult, sources: list[Chunk], start: float, cost: float | None
) -> GuardOut:
    kept = {id(r) for r in result.kept}
    return GuardOut(
        decision=str(result.decision),
        message=result.message,
        support_ratio=result.support_ratio,
        claims=[
            ClaimOut(
                text=r.claim.text,
                cited=r.claim.cited,
                verdict=str(r.verification.verdict),
                score=r.verification.score,
                kept=id(r) in kept,
                evidence_id=r.verification.evidence_id,
                rationale=r.verification.rationale,
            )
            for r in result.claims
        ],
        sources=[
            SourceOut(number=i, id=c.id, source=c.source, section=c.section, text=c.text)
            for i, c in enumerate(sources, start=1)
        ],
        latency_ms=(time.perf_counter() - start) * 1000,
        cost_usd=cost,
    )


def _total_cost(result: GuardResult, extra: float | None = 0.0) -> float | None:
    costs = [extra, *(r.verification.cost_usd for r in result.claims)]
    if any(c is None for c in costs):
        return None
    return sum(c or 0.0 for c in costs)


def _request_chunks(sources: list[SourceIn]) -> list[Chunk]:
    return [
        Chunk(
            id=s.id or f"source-{i}",
            source=s.id or f"source-{i}",
            section=s.title,
            start=0,
            end=len(s.text),
            text=s.text,
        )
        for i, s in enumerate(sources, start=1)
    ]


def create_app(
    verifier: Verifier,
    retriever: HybridRetriever | None = None,
    client: LLMClient | None = None,
) -> FastAPI:
    """Build the app. Without a retriever and client, /ask returns 503 and
    only /verify is available."""
    app = FastAPI(title="groundguard", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "verifier": verifier.name,
            "ask_enabled": retriever is not None and client is not None,
        }

    @app.post("/verify")
    def verify(request: VerifyRequest) -> GuardOut:
        start = time.perf_counter()
        chunks = _request_chunks(request.sources)
        every = list(range(1, len(chunks) + 1))
        if request.claims is not None:
            claims = [Claim(text=c.text, sources=c.sources) for c in request.claims]
        elif request.answer:
            claims = [
                Claim(text=request.answer[s:e], sources=every)
                for s, e in sentence_spans(request.answer)
            ]
        else:
            raise HTTPException(422, "send either `answer` or `claims`")
        answer = Answer("", chunks, StructuredAnswer(answerable=True, claims=claims), [])
        result = Guard(verifier, request.policy).check(answer)
        return _guard_out(result, chunks, start, _total_cost(result))

    @app.post("/ask")
    def ask(request: AskRequest) -> GuardOut:
        if retriever is None or client is None:
            raise HTTPException(503, "/ask needs an index and an LLM; only /verify is enabled")
        start = time.perf_counter()
        hits = retriever.search(request.question, k=request.k)
        chunks = [h.chunk for h in hits]
        answer = AnswerGenerator(client).generate(request.question, chunks)
        result = Guard(verifier, request.policy).check(answer)
        return _guard_out(result, chunks, start, _total_cost(result, answer.usage.cost_usd))

    return app


def components_from_env(
    index_dir: str = "data/index/k8s", verifier_kind: str = "nli"
) -> tuple[Verifier, HybridRetriever | None, LLMClient | None]:
    """Verifier, retriever and LLM client from disk and environment.

    The retriever is None if the index is missing and the client is None if
    LLM_BASE_URL/LLM_MODEL aren't set; callers then offer verification only.
    """
    from pathlib import Path

    from groundguard import corpus
    from groundguard.llm import LLMError, OpenAICompatibleClient
    from groundguard.verifiers import LLMJudgeVerifier, NLIVerifier

    client: LLMClient | None
    try:
        client = OpenAICompatibleClient.from_env()
    except LLMError:
        client = None

    chosen: Verifier
    if verifier_kind == "llm_judge":
        if client is None:
            raise RuntimeError("the llm_judge verifier needs LLM_BASE_URL and LLM_MODEL")
        chosen = LLMJudgeVerifier(client)
    else:
        chosen = NLIVerifier()

    index = Path(index_dir)
    retriever = corpus.load_retriever(index) if (index / "chunks.jsonl").exists() else None
    return chosen, retriever, client


def app_from_env(index_dir: str = "data/index/k8s", verifier_kind: str = "nli") -> FastAPI:
    """Wire the app for `groundguard serve`; see components_from_env."""
    return create_app(*components_from_env(index_dir, verifier_kind))
