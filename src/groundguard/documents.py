"""Loading source documents from disk.

A `Document` keeps the full original file text so that chunk offsets can be
checked against it later (`doc.text[chunk.start:chunk.end] == chunk.text`).
Front matter is not stripped from `text`; instead `body_start` records where
the body begins, and the chunker starts from there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_TITLE_LINE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    source: str
    """Path relative to the corpus root, with forward slashes."""
    text: str
    title: str
    body_start: int = 0


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def parse_markdown(source: str, text: str) -> Document:
    """Build a Document from Markdown, reading the title from YAML front matter.

    Only the `title:` key is read, with a regex rather than a YAML parser:
    Hugo front matter in the Kubernetes docs is simple, and pulling in a YAML
    dependency for one field isn't worth it.
    """
    title = ""
    body_start = 0
    match = _FRONT_MATTER.match(text)
    if match:
        body_start = match.end()
        title_match = _TITLE_LINE.search(match.group(1))
        if title_match:
            title = _strip_quotes(title_match.group(1))
    if not title:
        title = Path(source).stem.replace("-", " ").replace("_", " ")
    return Document(source=source, text=text, title=title, body_start=body_start)


def load_file(path: Path, root: Path) -> Document:
    source = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".md", ".markdown"}:
        return parse_markdown(source, text)
    return Document(source=source, text=text, title=path.stem)


def iter_documents(root: Path, patterns: Iterable[str] = ("**/*.md",)) -> Iterator[Document]:
    """Yield documents under `root` matching any glob pattern, in sorted order."""
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    for path in sorted(paths):
        yield load_file(path, root)
