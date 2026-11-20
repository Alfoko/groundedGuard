"""Split documents into retrievable chunks that remember where they came from.

Chunks never cross a Markdown section boundary, so every chunk has a single
section path ("Pods > Pod lifecycle > Restart policy"). Long sections are cut
into overlapping windows, preferring paragraph, then line, then sentence, then
word boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from groundguard.documents import Document

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
# Heading attributes such as "## Restart policy {#restart-policy}" (Hugo,
# Pandoc). They're anchors for links, not part of the title.
_HEADING_ATTRS = re.compile(r"\s*\{[#.][^}]*\}\s*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_BREAKS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Section:
    path: str
    start: int
    end: int


def _sections(doc: Document) -> list[_Section]:
    """Find section spans in the document body, ignoring headings inside code fences."""
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = []
    current_path = doc.title
    current_start = doc.body_start
    in_fence = False

    offset = doc.body_start
    for line in doc.text[doc.body_start :].splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if _FENCE.match(stripped):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING.match(stripped)
            if match:
                sections.append(_Section(current_path, current_start, offset))
                level = len(match.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                name = _HEADING_ATTRS.sub("", match.group(2)).strip()
                stack.append((level, name))
                current_path = " > ".join([doc.title, *(name for _, name in stack)])
                current_start = offset
        offset += len(line)
    sections.append(_Section(current_path, current_start, len(doc.text)))
    return [s for s in sections if doc.text[s.start : s.end].strip()]


def _cut_point(text: str, start: int, limit: int) -> int:
    """Pick an end offset in (start, limit] at the latest natural boundary."""
    window = text[start:limit]
    # Don't accept a boundary in the first half of the window, or we end up
    # producing lots of tiny chunks around dense line breaks.
    floor = len(window) // 2
    for sep in _BREAKS:
        idx = window.rfind(sep)
        if idx >= floor:
            return start + idx + len(sep)
    return limit


def _windows(
    text: str, start: int, end: int, max_chars: int, overlap: int
) -> Iterator[tuple[int, int]]:
    pos = start
    while pos < end:
        if end - pos <= max_chars:
            yield pos, end
            return
        cut = _cut_point(text, pos, pos + max_chars)
        yield pos, cut
        next_pos = max(cut - overlap, pos + 1)
        # Start the next window on a word boundary so it doesn't open mid-word.
        space = text.find(" ", next_pos, cut)
        pos = space + 1 if space != -1 else next_pos


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_document(doc: Document, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    chunks: list[Chunk] = []
    for section in _sections(doc):
        for raw_start, raw_end in _windows(
            doc.text, section.start, section.end, max_chars, overlap
        ):
            start, end = _trim(doc.text, raw_start, raw_end)
            if start == end:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc.source}#{len(chunks)}",
                    source=doc.source,
                    section=section.path,
                    start=start,
                    end=end,
                    text=doc.text[start:end],
                )
            )
    return chunks


def chunk_documents(
    docs: Iterable[Document], max_chars: int = 1200, overlap: int = 150
) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, max_chars=max_chars, overlap=overlap)
