"""The guard: verify every claim, then apply a policy to the answer.

Policies:
  strict  any claim that isn't supported rejects the whole answer
  drop    unsupported claims are removed and the rest is returned

Both decline when the supported share of claims falls below `min_support`,
or when nothing is left. Declining is a normal outcome, not an error: the
decline message says which claims the sources didn't back up, so the user
knows what's missing rather than just getting a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from groundguard.chunking import Chunk
from groundguard.citations import CheckedClaim, check_citations
from groundguard.generation import Answer
from groundguard.verifiers import NO_EVIDENCE, BatchVerifier, Verdict, Verification, Verifier

DECLINE_TEXT = "I can't answer that from the documents."


class PolicyMode(StrEnum):
    STRICT = "strict"
    DROP = "drop"


class GuardConfig(BaseModel):
    mode: PolicyMode = PolicyMode.DROP
    min_support: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum fraction of claims that must be supported to answer at all."""
    max_listed: int = Field(default=3, ge=0)
    """How many unsupported claims to name in a decline message."""


class Decision(StrEnum):
    ANSWER = "answer"
    PARTIAL = "partial"
    """Some claims were dropped; the rest is returned."""
    DECLINE = "decline"


@dataclass(frozen=True)
class ClaimResult:
    claim: CheckedClaim
    verification: Verification

    @property
    def supported(self) -> bool:
        return self.claim.citations_ok and self.verification.supported


@dataclass(frozen=True)
class GuardResult:
    decision: Decision
    claims: list[ClaimResult]
    message: str

    @property
    def kept(self) -> list[ClaimResult]:
        if self.decision is Decision.DECLINE:
            return []
        return [c for c in self.claims if c.supported]

    @property
    def dropped(self) -> list[ClaimResult]:
        if self.decision is Decision.DECLINE:
            return list(self.claims)
        return [c for c in self.claims if not c.supported]

    @property
    def support_ratio(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.supported for c in self.claims) / len(self.claims)


class Guard:
    def __init__(self, verifier: Verifier, config: GuardConfig | None = None) -> None:
        self.verifier = verifier
        self.config = config or GuardConfig()

    def verify_claims(self, answer: Answer) -> list[ClaimResult]:
        if isinstance(self.verifier, BatchVerifier):
            return self._verify_batch(self.verifier, answer)
        results = []
        for claim in check_citations(answer):
            if claim.citations_ok:
                verification = self.verifier.verify(claim.text, claim.evidence)
            else:
                # Nothing valid to check against; don't spend a verifier call.
                verification = NO_EVIDENCE
            results.append(ClaimResult(claim, verification))
        return results

    def _verify_batch(self, verifier: BatchVerifier, answer: Answer) -> list[ClaimResult]:
        checked = check_citations(answer)
        ok = [c for c in checked if c.citations_ok]
        evidence: list[Chunk] = []
        seen: set[str] = set()
        for claim in ok:
            for chunk in claim.evidence:
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    evidence.append(chunk)
        verdicts = iter(verifier.verify_batch([c.text for c in ok], evidence))
        return [ClaimResult(c, next(verdicts) if c.citations_ok else NO_EVIDENCE) for c in checked]

    def check(self, answer: Answer) -> GuardResult:
        results = self.verify_claims(answer)
        return apply_policy(results, self.config, answerable=answer.structured.answerable)


def apply_policy(
    results: list[ClaimResult], config: GuardConfig, *, answerable: bool = True
) -> GuardResult:
    if not answerable or not results:
        return GuardResult(Decision.DECLINE, results, decline_message(results, config))

    supported = [r for r in results if r.supported]
    ratio = len(supported) / len(results)
    if not supported or ratio < config.min_support:
        return GuardResult(Decision.DECLINE, results, decline_message(results, config))
    if len(supported) == len(results):
        return GuardResult(Decision.ANSWER, results, render(supported))
    if config.mode is PolicyMode.STRICT:
        return GuardResult(Decision.DECLINE, results, decline_message(results, config))
    return GuardResult(Decision.PARTIAL, results, render(supported))


def render(results: list[ClaimResult]) -> str:
    return " ".join(r.claim.text for r in results)


def _reason(result: ClaimResult) -> str:
    if not result.claim.cited:
        return "no source cited"
    if not result.claim.evidence:
        return "cited a source that wasn't provided"
    if result.verification.verdict is Verdict.CONTRADICTED:
        return "contradicted by the sources"
    return "not stated in the sources"


def decline_message(results: list[ClaimResult], config: GuardConfig) -> str:
    unsupported = [r for r in results if not r.supported]
    if not unsupported:
        return f"{DECLINE_TEXT} The retrieved sections don't cover this question."
    lines = [DECLINE_TEXT, "These points couldn't be confirmed:"]
    for result in unsupported[: config.max_listed]:
        lines.append(f"- {result.claim.text} ({_reason(result)})")
    hidden = len(unsupported) - config.max_listed
    if hidden > 0:
        lines.append(f"- and {hidden} more")
    return "\n".join(lines)
