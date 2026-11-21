import json
from collections.abc import Sequence

from groundguard.chunking import Chunk
from groundguard.generation import AnswerGenerator
from groundguard.guard import Decision, Guard
from groundguard.llm import FakeLLMClient
from groundguard.pipeline import GuardedAssistant
from groundguard.retrieval import BM25Index, HybridRetriever
from groundguard.verifiers import Verdict, Verification

CHUNKS = [
    Chunk("pods#0", "pods.md", "Pods", 0, 40, "A Pod is the smallest deployable unit."),
    Chunk("jobs#0", "jobs.md", "Jobs", 0, 40, "A Job retries failed Pods."),
]

REPLY = json.dumps(
    {
        "answerable": True,
        "claims": [
            {"text": "A Pod is the smallest deployable unit.", "sources": [1]},
            {"text": "Pods run on GPUs by default.", "sources": [1]},
        ],
    }
)


class SubstringVerifier:
    name = "substring"

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        hit = any(claim.rstrip(".") in c.text for c in evidence)
        return Verification(Verdict.SUPPORTED if hit else Verdict.NOT_ENOUGH_INFO, float(hit))


def assistant(guarded: bool) -> GuardedAssistant:
    return GuardedAssistant(
        HybridRetriever(BM25Index(CHUNKS)),
        AnswerGenerator(FakeLLMClient([REPLY])),
        Guard(SubstringVerifier()) if guarded else None,
        k=2,
    )


def test_guarded_pipeline_drops_unsupported_claim() -> None:
    response = assistant(guarded=True).ask("what is a pod")
    assert response.hits[0].chunk.id == "pods#0"
    assert response.guard is not None
    assert response.guard.decision is Decision.PARTIAL
    assert response.text == "A Pod is the smallest deployable unit."
    assert not response.declined


def test_unguarded_pipeline_returns_everything() -> None:
    response = assistant(guarded=False).ask("what is a pod")
    assert response.guard is None
    assert "GPUs" in response.text
