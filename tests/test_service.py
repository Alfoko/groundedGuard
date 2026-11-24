import json
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from groundguard.chunking import Chunk
from groundguard.llm import FakeLLMClient
from groundguard.retrieval import BM25Index, HybridRetriever
from groundguard.service import create_app
from groundguard.verifiers import Verdict, Verification


class WordVerifier:
    """Supported iff every word of the claim appears in some evidence chunk."""

    name = "words"

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        words = [w.strip(".,").lower() for w in claim.split()]
        for chunk in evidence:
            if all(w in chunk.text.lower() for w in words):
                return Verification(Verdict.SUPPORTED, 1.0, chunk.id)
        return Verification(Verdict.NOT_ENOUGH_INFO, 0.0)


SOURCES = [
    {"id": "pods", "text": "A Pod is the smallest deployable unit."},
    {"id": "jobs", "text": "A Job retries failed Pods up to backoffLimit times."},
]


def test_health_reports_verifier_and_mode() -> None:
    client = TestClient(create_app(WordVerifier()))
    assert client.get("/health").json() == {
        "status": "ok",
        "verifier": "words",
        "ask_enabled": False,
    }


def test_verify_free_text_answer() -> None:
    client = TestClient(create_app(WordVerifier()))
    body = {
        "sources": SOURCES,
        "answer": "A Pod is the smallest deployable unit. A Job retries forever.",
    }
    data = client.post("/verify", json=body).json()
    assert data["decision"] == "partial"
    assert data["message"] == "A Pod is the smallest deployable unit."
    assert [(c["verdict"], c["kept"]) for c in data["claims"]] == [
        ("supported", True),
        ("not_enough_info", False),
    ]
    assert data["claims"][0]["evidence_id"] == "pods"
    assert data["claims"][0]["cited"] == [1, 2]


def test_verify_structured_claims_and_strict_policy() -> None:
    client = TestClient(create_app(WordVerifier()))
    body = {
        "sources": SOURCES,
        "claims": [
            {"text": "A Pod is the smallest deployable unit.", "sources": [1]},
            {"text": "A Job retries failed Pods.", "sources": [1]},  # cites the wrong source
        ],
        "policy": {"mode": "strict"},
    }
    data = client.post("/verify", json=body).json()
    assert data["decision"] == "decline"
    assert "A Job retries failed Pods." in data["message"]


def test_verify_validates_input() -> None:
    client = TestClient(create_app(WordVerifier()))
    assert client.post("/verify", json={"sources": SOURCES}).status_code == 422
    assert client.post("/verify", json={"sources": [], "answer": "x"}).status_code == 422
    bad_policy = {"sources": SOURCES, "answer": "x.", "policy": {"min_support": 2}}
    assert client.post("/verify", json=bad_policy).status_code == 422


def test_ask_without_index_is_unavailable() -> None:
    client = TestClient(create_app(WordVerifier()))
    assert client.post("/ask", json={"question": "what is a pod"}).status_code == 503


def test_ask_runs_the_full_pipeline() -> None:
    chunks = [
        Chunk(s["id"], f"{s['id']}.md", s["id"].title(), 0, len(s["text"]), s["text"])
        for s in SOURCES
    ]
    reply = json.dumps(
        {
            "answerable": True,
            "claims": [
                {"text": "A Pod is the smallest deployable unit.", "sources": [1]},
                {"text": "Pods run on Mars.", "sources": [1]},
            ],
        }
    )
    llm = FakeLLMClient([reply], cost_per_call=0.002)
    app = create_app(WordVerifier(), HybridRetriever(BM25Index(chunks)), llm)
    data = TestClient(app).post("/ask", json={"question": "what is a pod", "k": 1}).json()

    assert data["decision"] == "partial"
    assert data["message"] == "A Pod is the smallest deployable unit."
    assert [s["id"] for s in data["sources"]] == ["pods"]
    assert data["cost_usd"] == pytest.approx(0.002)
    assert data["latency_ms"] >= 0


def test_app_from_env_starts_in_verify_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from groundguard.service import app_from_env

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    app = app_from_env(index_dir=str(tmp_path), verifier_kind="nli")
    health = TestClient(app).get("/health").json()
    assert health["ask_enabled"] is False
    assert health["verifier"].startswith("nli:")
