"""Claim verifiers: decide whether cited evidence supports a claim.

All verifiers share one small interface so the policy engine and the eval
harness don't care whether a verdict came from an NLI model, an LLM judge or
a stub in a test.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from groundguard.chunking import Chunk
from groundguard.llm import LLMClient, Message


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFO = "not_enough_info"


@dataclass(frozen=True)
class Verification:
    verdict: Verdict
    score: float
    """Confidence that the claim is supported, in [0, 1]."""
    evidence_id: str | None = None
    """The chunk that decided the verdict, when there is a single one."""
    rationale: str = ""
    cost_usd: float | None = 0.0
    """What producing this verdict cost; None if the provider didn't say."""

    @property
    def supported(self) -> bool:
        return self.verdict is Verdict.SUPPORTED


class Verifier(Protocol):
    name: str

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification: ...


@runtime_checkable
class BatchVerifier(Protocol):
    """A verifier that judges all of an answer's claims in one go."""

    name: str

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification: ...

    def verify_batch(
        self, claims: Sequence[str], evidence: Sequence[Chunk]
    ) -> list[Verification]: ...


NO_EVIDENCE = Verification(Verdict.NOT_ENOUGH_INFO, 0.0, rationale="no evidence cited")

# (premise, hypothesis) pairs -> one {"entailment", "neutral", "contradiction"}
# probability dict per pair.
NLIScorer = Callable[[Sequence[tuple[str, str]]], list[dict[str, float]]]


class NLIVerifier:
    """Entailment check with an off-the-shelf MNLI-style cross-encoder.

    Each cited chunk is scored on its own and the best-supporting chunk wins.
    That keeps premises inside the model's 512-token window, at the cost of
    missing claims that only follow from combining two chunks; such claims
    come out as not-enough-info rather than supported, which errs on the
    safe side for a guard.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        threshold: float = 0.5,
        scorer: NLIScorer | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = f"nli:{model_name}"
        self.threshold = threshold
        self._scorer = scorer
        self._load_lock = threading.Lock()

    def _score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        if self._scorer is None:
            # Eval workers share one verifier; without the lock each thread
            # that arrives before the first load finishes loads its own copy.
            with self._load_lock:
                if self._scorer is None:
                    self._scorer = _hf_scorer(self.model_name)
        return self._scorer(pairs)

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        if not evidence:
            return NO_EVIDENCE
        probs = self._score([(chunk.text, claim) for chunk in evidence])
        best = max(range(len(evidence)), key=lambda i: probs[i]["entailment"])
        entail = probs[best]["entailment"]
        if entail >= self.threshold:
            return Verification(Verdict.SUPPORTED, entail, evidence[best].id)
        # A contradiction only counts if some chunk actively contradicts the
        # claim more strongly than any chunk supports it.
        worst = max(range(len(evidence)), key=lambda i: probs[i]["contradiction"])
        if probs[worst]["contradiction"] > max(entail, probs[worst]["neutral"]):
            return Verification(Verdict.CONTRADICTED, entail, evidence[worst].id)
        return Verification(Verdict.NOT_ENOUGH_INFO, entail, evidence[best].id)


def _hf_scorer(model_name: str, batch_size: int = 16) -> NLIScorer:
    """Load a Hugging Face sequence-classification model (needs the `nli` extra)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Some NLI checkpoints ship fp16 weights; on CPU that's both slower and
    # less precise than fp32, and recent transformers keep the stored dtype.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    labels = {i: _normalise_label(name) for i, name in model.config.id2label.items()}

    def score(pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            enc: Any = tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
            out.extend({labels[i]: p for i, p in enumerate(row)} for row in probs)
        return out

    return score


def _normalise_label(name: str) -> str:
    name = name.lower()
    for label in ("entailment", "neutral", "contradiction"):
        if name.startswith(label[:5]):
            return label
    raise ValueError(f"unexpected NLI label {name!r}")


JUDGE_PROMPT = """\
You check whether a claim is supported by the given evidence passages. Judge \
only from the passages, not from what you know.

- "supported": the passages state the claim or directly imply it. Every part \
of the claim must be covered, including numbers, names and conditions.
- "contradicted": a passage says the opposite of the claim.
- "not_enough_info": anything else, including claims that are only partly \
covered.

Reply with a single JSON object and nothing else:
{"verdict": "supported", "passage": 1, "confidence": 0.9, "reason": "..."}

"passage" is the number of the passage your verdict rests on, or null. \
"confidence" is how sure you are of the verdict, from 0 to 1. Keep "reason" \
to one sentence."""


class _JudgeReply(BaseModel):
    verdict: Verdict
    passage: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class LLMJudgeVerifier:
    """Ask an LLM whether the cited passages support the claim.

    Unlike the NLI verifier the judge sees all cited passages at once, so it
    can accept claims that combine facts from two chunks. A reply that can't
    be parsed is treated as not-enough-info: for a guard, failing closed is
    the right default.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.name = f"llm-judge:{client.model}"

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        if not evidence:
            return NO_EVIDENCE
        passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(evidence, start=1))
        messages: list[Message] = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Passages:\n\n{passages}\n\nClaim: {claim}"},
        ]
        completion = self.client.complete(messages, json_mode=True)
        cost = completion.usage.cost_usd
        try:
            reply = _JudgeReply.model_validate_json(_strip_fence(completion.text))
        except ValidationError:
            return Verification(
                Verdict.NOT_ENOUGH_INFO,
                0.0,
                rationale="judge reply could not be parsed",
                cost_usd=cost,
            )
        evidence_id = None
        if reply.passage is not None and 1 <= reply.passage <= len(evidence):
            evidence_id = evidence[reply.passage - 1].id
        score = reply.confidence if reply.verdict is Verdict.SUPPORTED else 1 - reply.confidence
        return Verification(reply.verdict, score, evidence_id, reply.reason, cost)


def _strip_fence(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    return body.strip()


SELF_CHECK_PROMPT = """\
You are reviewing an answer that was written from the numbered source \
passages below. Find every numbered answer sentence that is not fully \
supported by the passages: anything contradicted, and anything added that \
the passages don't state, including details, numbers and conditions.

Reply with a single JSON object and nothing else:
{"unsupported": [2, 5]}

Use an empty list if every sentence is supported."""


class _SelfCheckReply(BaseModel):
    unsupported: list[int]


class SelfCheckVerifier:
    """Baseline: one LLM call reviews the whole answer at once.

    Much cheaper than judging claim by claim, since the sources are sent
    once per answer rather than once per claim, but the model has to keep
    every sentence in view at the same time. It's the "just ask the model
    to check its answer" approach the per-claim guard is compared against.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.name = f"self-check:{client.model}"

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        return self.verify_batch([claim], evidence)[0]

    def verify_batch(self, claims: Sequence[str], evidence: Sequence[Chunk]) -> list[Verification]:
        if not claims:
            return []
        if not evidence:
            return [NO_EVIDENCE] * len(claims)
        passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(evidence, start=1))
        sentences = "\n".join(f"{i}. {text}" for i, text in enumerate(claims, start=1))
        messages: list[Message] = [
            {"role": "system", "content": SELF_CHECK_PROMPT},
            {"role": "user", "content": f"Passages:\n\n{passages}\n\nAnswer:\n{sentences}"},
        ]
        completion = self.client.complete(messages, json_mode=True)
        total = completion.usage.cost_usd
        # Split the call's cost evenly so per-claim costs still add up.
        share = None if total is None else total / len(claims)
        try:
            reply = _SelfCheckReply.model_validate_json(_strip_fence(completion.text))
        except ValidationError:
            failed = Verification(
                Verdict.NOT_ENOUGH_INFO, 0.0, rationale="self-check reply could not be parsed"
            )
            return [replace(failed, cost_usd=share) for _ in claims]
        flagged = set(reply.unsupported)
        return [
            Verification(Verdict.NOT_ENOUGH_INFO, 0.0, cost_usd=share)
            if i in flagged
            else Verification(Verdict.SUPPORTED, 1.0, cost_usd=share)
            for i in range(1, len(claims) + 1)
        ]
