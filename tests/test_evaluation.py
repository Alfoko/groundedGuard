import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from groundguard import cli
from groundguard.chunking import Chunk
from groundguard.evaluation import (
    DatasetConfig,
    EvalConfig,
    ExampleResult,
    VerifierConfig,
    run_eval,
    sample_examples,
    write_report,
)
from groundguard.llm import FakeLLMClient
from groundguard.ragtruth import Example, Span
from groundguard.verifiers import LLMJudgeVerifier, Verdict, Verification

CONTEXT = "The cafe opens at 8am. It serves coffee and tea."


def example(i: int, response: str, spans: list[Span], split: str = "dev") -> Example:
    return Example(
        id=str(i),
        source_id=str(i),
        task_type="Summary",
        model="m",
        split=split,  # type: ignore[arg-type]
        quality="good",
        question="",
        context=CONTEXT,
        response=response,
        spans=spans,
    )


EXAMPLES = [
    # clean, two claims
    example(1, "The cafe opens at 8am. It serves coffee.", []),
    # second sentence hallucinated
    example(2, "The cafe opens at 8am. It closes at 9pm.", [Span(23, 40, "Evident Baseless Info")]),
    # both hallucinated
    example(3, "It opens at noon. It sells pizza.", [Span(0, 17, "c"), Span(18, 33, "b")]),
    example(4, "Not in the dev split.", [], split="test"),
]


class InContextVerifier:
    """Supported iff the claim's words all appear in the evidence."""

    name = "in-context"

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        text = " ".join(c.text for c in evidence).lower()
        words = [w.strip(".,").lower() for w in claim.split()]
        ok = all(w in text for w in words)
        return Verification(Verdict.SUPPORTED if ok else Verdict.NOT_ENOUGH_INFO, float(ok))


def config(tmp_path: Path, **kw: object) -> EvalConfig:
    return EvalConfig(name="t", output_dir=tmp_path, bootstrap_resamples=200, **kw)  # type: ignore[arg-type]


def test_baseline_shows_every_claim(tmp_path: Path) -> None:
    run = run_eval(config(tmp_path), examples=EXAMPLES)
    assert [r.example_id for r in run.results] == ["1", "2", "3"]
    assert sum(r.shown for r in run.results) == 6
    assert sum(r.shown_hallucinated for r in run.results) == 3


def test_guard_drops_hallucinated_claims(tmp_path: Path) -> None:
    cfg = config(tmp_path, verifier=VerifierConfig(kind="nli"))
    run = run_eval(cfg, examples=EXAMPLES, verifier=InContextVerifier())
    by_id = {r.example_id: r for r in run.results}

    assert by_id["1"].decision == "answer" and by_id["1"].shown == 2
    assert by_id["2"].decision == "partial"
    assert (by_id["2"].shown, by_id["2"].shown_hallucinated) == (1, 0)
    assert by_id["2"].caught == 1
    assert by_id["3"].decision == "decline" and by_id["3"].shown == 0


def test_report_files_and_metrics(tmp_path: Path) -> None:
    cfg = config(tmp_path, verifier=VerifierConfig(kind="nli"))
    out = write_report(run_eval(cfg, examples=EXAMPLES, verifier=InContextVerifier()))
    data = json.loads((out / "report.json").read_text())

    assert data["n"] == 3
    metrics = data["metrics"]
    assert metrics["hallucinated_claim_rate"]["value"] == 0.0
    assert metrics["over_refusal_rate"]["value"] == pytest.approx(1 / 3)
    assert metrics["verifier_recall"]["value"] == 1.0
    assert metrics["verifier_precision"]["value"] == 1.0
    assert "| over refusal rate | 33.3%" in (out / "report.md").read_text()
    assert len((out / "records.jsonl").read_text().splitlines()) == 3


def test_subset_sampling_is_seeded() -> None:
    many = [example(i, "x.", []) for i in range(50)]
    cfg = DatasetConfig(split="dev", n=10)
    first = [e.id for e in sample_examples(many, cfg, seed=3)]
    assert first == [e.id for e in sample_examples(many, cfg, seed=3)]
    assert first != [e.id for e in sample_examples(many, cfg, seed=4)]
    assert len(first) == 10


def test_budget_stops_the_run_cleanly(tmp_path: Path) -> None:
    reply = '{"verdict": "supported", "passage": 1, "confidence": 0.9}'
    client = FakeLLMClient(lambda _: reply, cost_per_call=0.01)
    cfg = config(tmp_path, verifier=VerifierConfig(kind="llm_judge"), budget_usd=0.025, workers=1)
    run = run_eval(cfg, examples=EXAMPLES, verifier=LLMJudgeVerifier(client), client=client)

    assert run.stopped_early is not None
    assert len(run.results) == 1  # 2 calls for example 1, budget hit during example 2
    assert run.results[0].cost_usd == pytest.approx(0.02)
    out = write_report(run)
    assert "Stopped early" in (out / "report.md").read_text()


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "name: judge-dev\nseed: 7\ndataset: {split: dev, n: 100}\n"
        "verifier: {kind: llm_judge}\npolicy: {mode: strict, min_support: 0.8}\n"
        "budget_usd: 5\n"
    )
    cfg = EvalConfig.from_yaml(path)
    assert cfg.dataset.n == 100
    assert cfg.verifier is not None and cfg.verifier.kind == "llm_judge"
    assert cfg.policy.mode == "strict"
    assert cfg.budget_usd == 5


def test_cli_eval_command_is_wired(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["eval", str(tmp_path / "c.yaml")])
    assert args.command == "eval"


def test_parallel_run_matches_sequential(tmp_path: Path) -> None:
    def run(workers: int) -> list[tuple[str, str, int]]:
        cfg = config(tmp_path, verifier=VerifierConfig(kind="nli"), workers=workers)
        results = run_eval(cfg, examples=EXAMPLES, verifier=InContextVerifier()).results
        return [(r.example_id, r.decision, r.shown) for r in results]

    assert run(4) == run(1)


def test_judge_cost_is_attributed_per_response(tmp_path: Path) -> None:
    reply = '{"verdict": "supported", "passage": 1, "confidence": 0.9}'
    client = FakeLLMClient(lambda _: reply, cost_per_call=0.01)
    cfg = config(tmp_path, verifier=VerifierConfig(kind="llm_judge"), workers=3)
    run = run_eval(cfg, examples=EXAMPLES, verifier=LLMJudgeVerifier(client), client=client)
    # One judge call per claim, two claims per dev response.
    assert [r.cost_usd for r in run.results] == pytest.approx([0.02, 0.02, 0.02])


class CountingVerifier(InContextVerifier):
    def __init__(self) -> None:
        self.claims: list[str] = []

    def verify(self, claim: str, evidence: Sequence[Chunk]) -> Verification:
        self.claims.append(claim)
        return super().verify(claim, evidence)


def test_interrupted_run_resumes_from_progress(tmp_path: Path) -> None:
    from groundguard.evaluation import progress_path

    cfg = config(tmp_path, verifier=VerifierConfig(kind="nli"))
    full = run_eval(cfg, examples=EXAMPLES, verifier=InContextVerifier(), resume=False)

    # Simulate a kill after the first response, mid-way through writing the second.
    path = progress_path(cfg)
    header, first, second = path.read_text().splitlines()[:3]
    path.write_text(f"{header}\n{first}\n{second[:20]}")

    counting = CountingVerifier()
    resumed = run_eval(cfg.model_copy(update={"workers": 3}), examples=EXAMPLES, verifier=counting)

    def key(r: ExampleResult) -> tuple[str, str, int, int]:
        return (r.example_id, r.decision, r.shown, r.caught)

    assert [key(r) for r in resumed.results] == [key(r) for r in full.results]
    assert len(counting.claims) == 4  # only the two unfinished responses were checked
    assert len(path.read_text().splitlines()) == 4

    write_report(resumed)
    assert not path.exists()


def test_changed_config_ignores_old_progress(tmp_path: Path) -> None:
    cfg = config(tmp_path, verifier=VerifierConfig(kind="nli"))
    run_eval(cfg, examples=EXAMPLES, verifier=InContextVerifier())
    counting = CountingVerifier()
    stricter = cfg.model_copy(update={"policy": cfg.policy.model_copy(update={"mode": "strict"})})
    run_eval(stricter, examples=EXAMPLES, verifier=counting)
    assert len(counting.claims) == 6
