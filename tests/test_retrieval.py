import hashlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from groundguard.chunking import Chunk
from groundguard.retrieval import (
    BM25Index,
    DenseIndex,
    HybridRetriever,
    ScoredChunk,
    reciprocal_rank_fusion,
    tokenize,
)


def make_chunk(cid: str, text: str, section: str = "Doc") -> Chunk:
    return Chunk(id=cid, source="doc.md", section=section, start=0, end=len(text), text=text)


CHUNKS = [
    make_chunk("pods", "A Pod is a group of one or more containers with shared storage."),
    make_chunk("svc", "A Service exposes an application running on a set of Pods."),
    make_chunk("cron", "A CronJob creates Jobs on a repeating schedule."),
    make_chunk("node", "A node is a worker machine; the kubelet runs on every node."),
]


class BagOfWordsEmbedder:
    """Deterministic stand-in for a sentence encoder: hashed token counts."""

    dims = 64

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims
                out[row, bucket] += 1.0
        return out


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("What is the kube-proxy?") == ["kube", "proxy"]


def test_tokenize_folds_simple_plurals() -> None:
    assert tokenize("Pods pod nodes class") == ["pod", "pod", "node", "class"]


def test_bm25_ranks_matching_chunk_first() -> None:
    index = BM25Index(CHUNKS)
    results = index.search("how does a cronjob schedule jobs", k=2)
    assert results[0].chunk.id == "cron"
    assert all(r.score > 0 for r in results)


def test_bm25_rare_terms_outweigh_common_ones() -> None:
    index = BM25Index(CHUNKS)
    # "pods" appears in two chunks, "storage" in one.
    assert index.search("pods storage")[0].chunk.id == "pods"


def test_bm25_matches_section_path() -> None:
    chunks = [*CHUNKS, make_chunk("rp", "Applies to all containers.", section="Restart policy")]
    assert BM25Index(chunks).search("restart policy")[0].chunk.id == "rp"


def test_bm25_unknown_terms_return_nothing() -> None:
    assert BM25Index(CHUNKS).search("zebra") == []
    assert BM25Index([]).search("pods") == []


def test_dense_index_returns_nearest_neighbour() -> None:
    index = DenseIndex(CHUNKS, BagOfWordsEmbedder())
    top = index.search("kubelet worker machine", k=1)
    assert top[0].chunk.id == "node"
    assert 0 < top[0].score <= 1.0 + 1e-6  # cosine similarity of unit vectors


def test_dense_index_rejects_mismatched_vectors() -> None:
    with pytest.raises(ValueError):
        DenseIndex(CHUNKS, BagOfWordsEmbedder(), vectors=np.zeros((2, 4), dtype=np.float32))


def test_rrf_rewards_agreement_between_rankings() -> None:
    a, b, c = (make_chunk(x, x) for x in "abc")
    first = [ScoredChunk(a, 9.0), ScoredChunk(b, 5.0)]
    second = [ScoredChunk(c, 0.9), ScoredChunk(b, 0.8)]
    fused = reciprocal_rank_fusion([first, second], k=60)
    assert [s.chunk.id for s in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(2 / 62)


def test_hybrid_without_dense_matches_bm25_order() -> None:
    bm25 = BM25Index(CHUNKS)
    hybrid = HybridRetriever(bm25)
    expected = [r.chunk.id for r in bm25.search("pods service", k=3)]
    assert [r.chunk.id for r in hybrid.search("pods service", k=3)] == expected


def test_hybrid_uses_dense_hits_bm25_misses() -> None:
    hybrid = HybridRetriever(BM25Index(CHUNKS), DenseIndex(CHUNKS, BagOfWordsEmbedder()))
    ids = [r.chunk.id for r in hybrid.search("kubelet", k=4)]
    assert ids[0] == "node"
    assert len(ids) == 4  # dense ranking contributes the chunks BM25 scored zero
