"""Hybrid retrieval: BM25 over tokens plus optional dense embeddings.

The two rankings are merged with reciprocal rank fusion (Cormack et al.,
2009). RRF only looks at ranks, so there's no need to calibrate BM25 scores
against cosine similarities, which live on unrelated scales.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from groundguard.chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


def _fold_plural(token: str) -> str:
    # Deliberately crude: it only needs to make "pods" and "pod" meet. A full
    # stemmer would also merge things like "ingress" and "ingresses" but adds
    # a dependency and mangles Kubernetes identifiers.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_fold_plural(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 with the usual k1/b parameters and Lucene-style IDF."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tfs = [Counter(tokenize(_index_text(c))) for c in self.chunks]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self.chunks)
        # The +1 inside the log keeps IDF positive for very common terms.
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        if not terms:
            return []
        scored: list[ScoredChunk] = []
        for chunk, tf, length in zip(self.chunks, self._tfs, self._lengths, strict=True):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scored.append(ScoredChunk(chunk, score))
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[:k]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Return one row per text."""
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by sentence-transformers (install the `embeddings` extra)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        vectors = self._model.encode(list(texts), batch_size=64, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def _normalize(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: npt.NDArray[np.float32] = (matrix / norms).astype(np.float32)
    return normalized


def _index_text(chunk: Chunk) -> str:
    # The section path often names the topic better than the body does
    # ("Pods > Restart policy"), so it's indexed along with the text.
    return f"{chunk.section}\n{chunk.text}"


class DenseIndex:
    """Brute-force cosine search over an in-memory matrix.

    The demo corpus is a few tens of thousands of chunks, where a matrix
    product is fast enough and avoids an ANN dependency.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        vectors: npt.NDArray[np.float32] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        if vectors is None:
            vectors = embedder.embed([_index_text(c) for c in self.chunks])
        if vectors.shape[0] != len(self.chunks):
            raise ValueError("need exactly one vector per chunk")
        self.vectors = _normalize(vectors)

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q = _normalize(self.embedder.embed([query]))[0]
        sims = self.vectors @ q
        top = np.argsort(-sims, kind="stable")[:k]
        return [ScoredChunk(self.chunks[i], float(sims[i])) for i in top]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[item.chunk.id] = item.chunk
    out = [ScoredChunk(by_id[cid], score) for cid, score in fused.items()]
    out.sort(key=lambda s: (-s.score, s.chunk.id))
    return out


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex | None = None,
        rrf_k: int = 60,
        candidates: int = 50,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        rankings = [self.bm25.search(query, self.candidates)]
        if self.dense is not None:
            rankings.append(self.dense.search(query, self.candidates))
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
