"""Claim verifiers: decide whether cited evidence supports a claim.

All verifiers share one small interface so the policy engine and the eval
harness don't care whether a verdict came from an NLI model, an LLM judge or
a stub in a test.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from groundguard.chunking import Chunk


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

    @property
    def supported(self) -> bool:
        return self.verdict is Verdict.SUPPORTED


class Verifier(Protocol):
    name: str

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification: ...


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

    def _score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
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
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
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
