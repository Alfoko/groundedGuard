"""Turn a generated answer into checkable claims and validate their citations.

This is the cheap first pass of the guard. It needs no model: a claim that
cites nothing, or cites only sources the model was never shown, can't be
grounded no matter what the verifier would say, so it's flagged here before
any verification cost is spent on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from groundguard.chunking import Chunk
from groundguard.generation import Answer


class CitationStatus(StrEnum):
    OK = "ok"
    NO_CITATION = "no_citation"
    INVALID_ONLY = "invalid_only"
    """Every cited number is out of range."""


@dataclass(frozen=True)
class CheckedClaim:
    index: int
    text: str
    cited: list[int]
    evidence: list[Chunk] = field(default_factory=list)
    """The valid cited chunks, de-duplicated, in citation order."""
    invalid: list[int] = field(default_factory=list)

    @property
    def status(self) -> CitationStatus:
        if not self.cited:
            return CitationStatus.NO_CITATION
        if not self.evidence:
            return CitationStatus.INVALID_ONLY
        return CitationStatus.OK

    @property
    def citations_ok(self) -> bool:
        return self.status is CitationStatus.OK


def check_citations(answer: Answer) -> list[CheckedClaim]:
    """Resolve each claim's source numbers to chunks.

    A claim with a mix of valid and invalid numbers keeps its valid evidence
    and still passes; the invalid numbers are recorded so they can be
    reported, but dropping a claim over one bad reference would be harsher
    than the verifier step needs it to be.
    """
    checked: list[CheckedClaim] = []
    for index, claim in enumerate(answer.claims):
        evidence: list[Chunk] = []
        seen: set[str] = set()
        invalid: list[int] = []
        for number in claim.sources:
            chunk = answer.chunk_for(number)
            if chunk is None:
                if number not in invalid:
                    invalid.append(number)
            elif chunk.id not in seen:
                seen.add(chunk.id)
                evidence.append(chunk)
        checked.append(
            CheckedClaim(
                index=index,
                text=claim.text.strip(),
                cited=list(claim.sources),
                evidence=evidence,
                invalid=invalid,
            )
        )
    return checked
