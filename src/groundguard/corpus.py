"""Fetching the Kubernetes docs and building a persisted retrieval index.

The index is a directory with three files:
  chunks.jsonl   one chunk per line
  meta.json      corpus revision and chunking settings
  vectors.npy    dense embeddings (only if built with an embedder)
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from groundguard.chunking import Chunk, chunk_documents
from groundguard.documents import Document, parse_markdown
from groundguard.retrieval import BM25Index, DenseIndex, Embedder, HybridRetriever

K8S_REPO = "https://github.com/kubernetes/website"
# kubernetes/website main as of 2026-11-20. Pinned so the corpus, and anything
# measured against it, can be reproduced.
K8S_COMMIT = "be58364903ad78d60314019b81759f05c33b4af5"
K8S_DOCS_PATH = "content/en/docs"

_TOOLTIP = re.compile(r"\{\{<\s*glossary_tooltip\b[^>]*?\btext=\"([^\"]*)\"[^>]*>\}\}")
_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_hugo(text: str) -> str:
    """Strip Hugo shortcodes and comments, keeping the visible text of tooltips.

    Chunk offsets refer to this cleaned text, not the raw file, since that's
    what gets indexed and shown to the model.
    """
    text = _TOOLTIP.sub(r"\1", text)
    text = _SHORTCODE.sub("", text)
    return _HTML_COMMENT.sub("", text)


def fetch_kubernetes_docs(dest: Path, commit: str = K8S_COMMIT) -> Path:
    """Sparse, shallow checkout of the English docs at a fixed commit.

    Returns the docs root. Re-running against an existing checkout just moves
    it to `commit`.
    """
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dest), *args], check=True)

    if not (dest / ".git").exists():
        git("init", "-q")
        git("remote", "add", "origin", K8S_REPO)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("sparse-checkout", "set", K8S_DOCS_PATH)
    git("checkout", "-q", "FETCH_HEAD")
    return dest / K8S_DOCS_PATH


def load_hugo_docs(root: Path) -> list[Document]:
    docs = []
    for path in sorted(root.rglob("*.md")):
        source = path.relative_to(root).as_posix()
        doc = parse_markdown(source, clean_hugo(path.read_text(encoding="utf-8")))
        if doc.text[doc.body_start :].strip():
            docs.append(doc)
    return docs


def save_index(
    out_dir: Path,
    chunks: list[Chunk],
    meta: dict[str, Any],
    embedder: Embedder | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    if embedder is not None:
        np.save(out_dir / "vectors.npy", DenseIndex(chunks, embedder).vectors)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def build_index(
    docs: Iterable[Document],
    out_dir: Path,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
    embedder: Embedder | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    chunks = list(chunk_documents(docs, max_chars=max_chars, overlap=overlap))
    meta = {
        "max_chars": max_chars,
        "overlap": overlap,
        "num_chunks": len(chunks),
        "embedder": getattr(embedder, "model_name", None) if embedder else None,
        **(extra_meta or {}),
    }
    save_index(out_dir, chunks, meta, embedder)
    return chunks


def load_chunks(index_dir: Path) -> list[Chunk]:
    with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def load_retriever(index_dir: Path, embedder: Embedder | None = None) -> HybridRetriever:
    """Load a saved index. Dense search is used only if vectors were saved
    and an embedder is supplied to encode queries."""
    chunks = load_chunks(index_dir)
    dense = None
    vectors_path = index_dir / "vectors.npy"
    if embedder is not None and vectors_path.exists():
        dense = DenseIndex(chunks, embedder, vectors=np.load(vectors_path))
    return HybridRetriever(BM25Index(chunks), dense)
