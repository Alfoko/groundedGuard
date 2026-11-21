"""End-to-end cases: retrieval, a canned model reply, scripted verdicts, the guard.

Each case lives in tests/golden/<name>.json with its expected outcome next to
it in <name>.expected.json. After an intentional behaviour change, rewrite
the expected files with:

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py

and review the diff before committing.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from groundguard.chunking import Chunk
from groundguard.generation import AnswerGenerator
from groundguard.guard import Guard, GuardConfig
from groundguard.llm import FakeLLMClient
from groundguard.pipeline import GuardedAssistant, GuardedResponse
from groundguard.retrieval import BM25Index, HybridRetriever
from groundguard.verifiers import Verdict, Verification

GOLDEN_DIR = Path(__file__).parent / "golden"
CASES = sorted(p for p in GOLDEN_DIR.glob("*.json") if not p.name.endswith(".expected.json"))


class StubVerifier:
    """Verdicts come from the fixture; unknown claims are not supported."""

    name = "stub"

    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        verdict = Verdict(self.verdicts.get(claim, Verdict.NOT_ENOUGH_INFO))
        return Verification(verdict, 1.0 if verdict is Verdict.SUPPORTED else 0.0, evidence[0].id)


def run_case(case: dict[str, Any]) -> GuardedResponse:
    chunks = [Chunk(start=0, end=len(c["text"]), **c) for c in case["chunks"]]
    assistant = GuardedAssistant(
        HybridRetriever(BM25Index(chunks)),
        AnswerGenerator(FakeLLMClient([case["reply"]])),
        Guard(StubVerifier(case["verdicts"]), GuardConfig(**case["config"])),
        k=len(chunks),
    )
    return assistant.ask(case["question"])


def summarise(response: GuardedResponse) -> dict[str, Any]:
    assert response.guard is not None
    return {
        "retrieved": [h.chunk.id for h in response.hits],
        "decision": str(response.guard.decision),
        "claims": [
            {
                "text": r.claim.text,
                "evidence": [c.id for c in r.claim.evidence],
                "invalid_citations": r.claim.invalid,
                "verdict": str(r.verification.verdict),
                "kept": r in response.guard.kept,
            }
            for r in response.guard.claims
        ],
        "message": response.text,
    }


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_golden_case(path: Path) -> None:
    actual = summarise(run_case(json.loads(path.read_text())))
    expected_path = path.with_suffix(".expected.json")
    if os.environ.get("UPDATE_GOLDEN"):
        expected_path.write_text(json.dumps(actual, indent=2) + "\n")
    assert expected_path.exists(), f"missing {expected_path.name}; run with UPDATE_GOLDEN=1"
    assert actual == json.loads(expected_path.read_text())


def test_golden_cases_exist() -> None:
    assert len(CASES) >= 5
