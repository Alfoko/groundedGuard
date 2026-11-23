"""RAGTruth loading, splits, and claim-level alignment to gold spans.

RAGTruth (Niu et al., 2024) has ~18k model responses over 2,965 source
documents, with character-level spans marking hallucinated text. We score at
the claim level: each response is cut into sentence-sized claims, and a claim
counts as hallucinated if it overlaps any gold span.

Splits:
  test   the official test split, used only for reported numbers
  dev    a held-out slice of the official train split, for tuning
  train  the rest of the official train split

The dev slice is assigned per source document, not per response. Several
models answered each source, and splitting their responses across train and
dev would leak the source text into both.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Split = Literal["train", "dev", "test"]

RAGTRUTH_REPO = "https://github.com/ParticleMedia/RAGTruth"
RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label_type: str

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True)
class Example:
    id: str
    source_id: str
    task_type: str
    model: str
    split: Split
    quality: str
    question: str
    """The user question for QA; empty for summary and data-to-text."""
    context: str
    response: str
    spans: list[Span] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return bool(self.spans)


@dataclass(frozen=True)
class GoldClaim:
    text: str
    start: int
    end: int
    hallucinated: bool
    label_types: list[str]


def _context(task_type: str, info: Any) -> tuple[str, str]:
    if task_type == "QA":
        return str(info["question"]), str(info["passages"]).strip()
    if isinstance(info, str):
        return "", info
    # Data2txt sources are structured business records; the models saw them
    # as JSON, so the verifier gets the same text.
    return "", json.dumps(info, ensure_ascii=False)


def dev_bucket(source_id: str, dev_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < dev_fraction


def load_ragtruth(root: Path, dev_fraction: float = 0.1, seed: int = 13) -> list[Example]:
    """Load the dataset from a checkout of the RAGTruth repository."""
    sources: dict[str, dict[str, Any]] = {}
    with (root / "dataset" / "source_info.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sources[str(row["source_id"])] = row

    examples = []
    with (root / "dataset" / "response.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source = sources[str(row["source_id"])]
            question, context = _context(source["task_type"], source["source_info"])
            split: Split = "test" if row["split"] == "test" else "train"
            if split == "train" and dev_bucket(str(row["source_id"]), dev_fraction, seed):
                split = "dev"
            examples.append(
                Example(
                    id=str(row["id"]),
                    source_id=str(row["source_id"]),
                    task_type=source["task_type"],
                    model=row["model"],
                    split=split,
                    quality=row.get("quality", "good"),
                    question=question,
                    context=context,
                    response=row["response"],
                    spans=[
                        Span(int(s["start"]), int(s["end"]), s["label_type"]) for s in row["labels"]
                    ],
                )
            )
    return examples


def select(examples: Iterable[Example], split: Split) -> list[Example]:
    return [e for e in examples if e.split == split]


_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
        "co",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "no",
        "approx",
    ]
)
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentence spans (start, end), trimmed of whitespace.

    Rule-based on purpose: model outputs in RAGTruth are short and plain, and
    a deterministic splitter keeps claim boundaries reproducible.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start()
        before = text[start:end].rstrip()
        last_word = before.rsplit(None, 1)[-1].rstrip(".").lower() if before else ""
        is_initial = len(last_word) == 1 and last_word.isalpha()
        if "\n" not in match.group() and (last_word in _ABBREVIATIONS or is_initial):
            continue
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    return [t for t in (_trim(text, s, e) for s, e in spans) if t[0] < t[1]]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def label_claims(claim_spans: Sequence[tuple[int, int]], gold: Sequence[Span]) -> list[list[str]]:
    """For each claim span, the label types of gold spans overlapping it."""
    return [
        sorted({g.label_type for g in gold if g.overlaps(start, end)}) for start, end in claim_spans
    ]


def gold_claims(example: Example) -> list[GoldClaim]:
    spans = sentence_spans(example.response)
    labels = label_claims(spans, example.spans)
    return [
        GoldClaim(example.response[s:e], s, e, bool(types), types)
        for (s, e), types in zip(spans, labels, strict=True)
    ]
