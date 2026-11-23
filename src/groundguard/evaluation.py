"""Eval runner: one YAML config in, per-answer records and a report out.

On RAGTruth the guard is applied to the dataset's own responses: each
response is split into sentence claims (see ragtruth.gold_claims), every
claim is checked against the source context, and the policy decides what
the user would have seen. Since RAGTruth responses carry no citations, each
claim is checked against all chunks of its source context.

Every RAGTruth question is answerable from its context by construction, so
over-refusal here is simply the share of responses the guard declined.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from groundguard.cache import CachedLLMClient
from groundguard.chunking import Chunk, chunk_document
from groundguard.documents import Document
from groundguard.generation import Answer, Claim, StructuredAnswer
from groundguard.guard import Decision, Guard, GuardConfig
from groundguard.llm import BudgetExceeded, LLMClient, OpenAICompatibleClient, UsageTracker
from groundguard.metrics import AnswerRecord, summarise
from groundguard.ragtruth import Example, Split, gold_claims, load_ragtruth, select
from groundguard.stats import Estimate, bootstrap_ci
from groundguard.verifiers import LLMJudgeVerifier, NLIVerifier, SelfCheckVerifier, Verifier


class DatasetConfig(BaseModel):
    root: Path = Path("data/raw/RAGTruth")
    split: Split = "dev"
    n: int | None = Field(default=None, ge=1)
    """Evaluate a seeded random subset of this size; None means all."""
    task_types: list[str] | None = None


class VerifierConfig(BaseModel):
    kind: Literal["nli", "llm_judge", "self_check"]
    model: str | None = None
    """NLI model name; for llm_judge the LLM_MODEL env var is used."""
    threshold: float = 0.5


class EvalConfig(BaseModel):
    name: str
    seed: int = 0
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    verifier: VerifierConfig | None = None
    """None evaluates the unguarded responses (the baseline)."""
    policy: GuardConfig = Field(default_factory=GuardConfig)
    budget_usd: float | None = None
    cache_path: Path = Path(".cache/llm.sqlite")
    output_dir: Path = Path("results")
    chunk_chars: int = 1200
    workers: int = Field(default=8, ge=1)
    """Responses evaluated in parallel; LLM-judge runs are I/O bound."""
    bootstrap_resamples: int = 10_000

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    task_type: str
    decision: str
    claims: int
    shown: int
    shown_hallucinated: int
    gold_hallucinated: int
    flagged: int
    caught: int
    """Gold-hallucinated claims the verifier flagged."""
    latency_s: float
    cost_usd: float | None

    def answer_record(self) -> AnswerRecord:
        return AnswerRecord(
            answerable=True,
            declined=self.decision == Decision.DECLINE,
            shown_claims=self.shown,
            shown_hallucinated=self.shown_hallucinated,
            latency_s=self.latency_s,
            cost_usd=self.cost_usd,
        )


def sample_examples(examples: list[Example], config: DatasetConfig, seed: int) -> list[Example]:
    pool = select(examples, config.split)
    if config.task_types:
        pool = [e for e in pool if e.task_type in config.task_types]
    pool.sort(key=lambda e: e.id)
    if config.n is not None and config.n < len(pool):
        pool = sorted(random.Random(seed).sample(pool, config.n), key=lambda e: e.id)
    return pool


def context_chunks(example: Example, max_chars: int) -> list[Chunk]:
    doc = Document(source=example.id, text=example.context, title=example.task_type)
    return chunk_document(doc, max_chars=max_chars, overlap=min(150, max_chars // 4))


def as_answer(example: Example, chunks: list[Chunk]) -> Answer:
    """Present a RAGTruth response to the guard as claims citing every chunk."""
    every_source = list(range(1, len(chunks) + 1))
    claims = [Claim(text=g.text, sources=every_source) for g in gold_claims(example)]
    structured = StructuredAnswer(answerable=True, claims=claims)
    return Answer(example.question, chunks, structured, [])


def evaluate_example(example: Example, guard: Guard | None, chunk_chars: int) -> ExampleResult:
    gold = gold_claims(example)
    labels = [g.hallucinated for g in gold]
    if guard is None:
        bad = sum(labels)
        return ExampleResult(
            example.id,
            example.task_type,
            str(Decision.ANSWER),
            len(gold),
            len(gold),
            bad,
            bad,
            0,
            0,
            0.0,
            0.0,
        )

    start = time.perf_counter()
    answer = as_answer(example, context_chunks(example, chunk_chars))
    result = guard.check(answer)
    latency = time.perf_counter() - start

    flagged = [not r.supported for r in result.claims]
    kept = {id(r) for r in result.kept}
    shown = [r for r in result.claims if id(r) in kept]
    shown_bad = sum(lab for r, lab in zip(result.claims, labels, strict=True) if id(r) in kept)
    costs = [r.verification.cost_usd for r in result.claims]
    return ExampleResult(
        example_id=example.id,
        task_type=example.task_type,
        decision=str(result.decision),
        claims=len(gold),
        shown=len(shown),
        shown_hallucinated=shown_bad,
        gold_hallucinated=sum(labels),
        flagged=sum(flagged),
        caught=sum(f and lab for f, lab in zip(flagged, labels, strict=True)),
        latency_s=latency,
        cost_usd=None if any(c is None for c in costs) else sum(c or 0.0 for c in costs),
    )


def build_verifier(config: VerifierConfig, client: LLMClient | None) -> Verifier:
    if config.kind == "nli":
        if config.model:
            return NLIVerifier(config.model, threshold=config.threshold)
        return NLIVerifier(threshold=config.threshold)
    if client is None:
        raise ValueError(f"{config.kind} needs an LLM client")
    if config.kind == "self_check":
        return SelfCheckVerifier(client)
    return LLMJudgeVerifier(client)


@dataclass
class EvalRun:
    config: EvalConfig
    results: list[ExampleResult]
    stopped_early: str | None = None


def run_eval(
    config: EvalConfig,
    *,
    examples: list[Example] | None = None,
    verifier: Verifier | None = None,
    client: LLMClient | None = None,
) -> EvalRun:
    """Run one configuration. `examples`, `verifier` and `client` can be
    injected for tests; otherwise they're built from the config."""
    if examples is None:
        examples = load_ragtruth(config.dataset.root, seed=config.seed)
    chosen = sample_examples(examples, config.dataset, config.seed)

    tracker = UsageTracker(budget_usd=config.budget_usd)
    guard = None
    if config.verifier is not None:
        if verifier is None:
            if client is None and config.verifier.kind != "nli":
                client = CachedLLMClient(
                    OpenAICompatibleClient.from_env(tracker=tracker), config.cache_path
                )
            verifier = build_verifier(config.verifier, client)
        elif client is not None and hasattr(client, "tracker"):
            tracker = client.tracker
            tracker.budget_usd = config.budget_usd
        guard = Guard(verifier, config.policy)

    run = EvalRun(config, [])
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [
            pool.submit(evaluate_example, example, guard, config.chunk_chars) for example in chosen
        ]
        for future in futures:
            try:
                run.results.append(future.result())
            except BudgetExceeded as exc:
                if run.stopped_early is None:
                    run.stopped_early = f"budget reached: {exc}"
                    for pending in futures:
                        pending.cancel()
            except CancelledError:
                continue
    return run


def _ci(num: list[int], den: list[int], config: EvalConfig) -> Estimate:
    return bootstrap_ci(num, den, n_resamples=config.bootstrap_resamples, seed=config.seed)


def report_data(run: EvalRun) -> dict[str, Any]:
    results, config = run.results, run.config
    records = [r.answer_record() for r in results]
    metrics = summarise(records)
    ones = [1] * len(results)
    rates = {
        "hallucinated_claim_rate": _ci(
            [r.shown_hallucinated for r in results], [r.shown for r in results], config
        ),
        "answer_hallucination_rate": _ci(
            [int(r.shown_hallucinated > 0 and r.decision != Decision.DECLINE) for r in results],
            [int(r.decision != Decision.DECLINE) for r in results],
            config,
        ),
        "over_refusal_rate": _ci(
            [int(r.decision == Decision.DECLINE) for r in results], ones, config
        ),
    }
    if config.verifier is not None:
        rates["verifier_precision"] = _ci(
            [r.caught for r in results], [r.flagged for r in results], config
        )
        rates["verifier_recall"] = _ci(
            [r.caught for r in results], [r.gold_hallucinated for r in results], config
        )
    return {
        "name": config.name,
        "config": json.loads(config.model_dump_json()),
        "n": metrics.n,
        "stopped_early": run.stopped_early,
        "metrics": {k: asdict(v) for k, v in rates.items()},
        "latency_p50_s": metrics.latency_p50_s,
        "latency_p95_s": metrics.latency_p95_s,
        "cost_per_answer_usd": metrics.cost_per_answer_usd,
        "total_cost_usd": sum(r.cost_usd or 0.0 for r in results),
    }


def _pct(e: dict[str, float]) -> str:
    return f"{e['value'] * 100:.1f}% [{e['low'] * 100:.1f}, {e['high'] * 100:.1f}]"


def render_markdown(data: dict[str, Any]) -> str:
    split = data["config"]["dataset"]["split"]
    lines = [
        f"# {data['name']}",
        "",
        f"RAGTruth `{split}` split, n = {data['n']} responses, "
        f"seed {data['config']['seed']}. Intervals are 95% bootstrap CIs over responses.",
        "",
    ]
    if data["stopped_early"]:
        lines += [f"**Stopped early:** {data['stopped_early']}", ""]
    lines += ["| Metric | Value |", "|---|---|"]
    for key, est in data["metrics"].items():
        lines.append(f"| {key.replace('_', ' ')} | {_pct(est)} |")
    lines.append(
        f"| latency p50 / p95 | {data['latency_p50_s']:.3f} s / {data['latency_p95_s']:.3f} s |"
    )
    cost = data["cost_per_answer_usd"]
    lines.append(f"| cost per answer | {'unknown' if cost is None else f'${cost:.5f}'} |")
    return "\n".join(lines) + "\n"


def write_report(run: EvalRun) -> Path:
    out = run.config.output_dir / run.config.name
    out.mkdir(parents=True, exist_ok=True)
    data = report_data(run)
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "report.md").write_text(render_markdown(data))
    with (out / "records.jsonl").open("w") as f:
        for result in run.results:
            f.write(json.dumps(asdict(result)) + "\n")
    return out
