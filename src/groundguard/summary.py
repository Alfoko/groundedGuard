"""Compare finished eval runs: one table plus paired tests against a reference.

Reads the records.jsonl files written by `groundguard eval`, so the summary
is built only from committed run output. Runs are compared on the responses
they have in common; with the configs in configs/ragtruth that is the same
seeded sample for every system.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundguard.stats import Comparison, Estimate, bootstrap_ci, paired_permutation_test

Pair = tuple[list[int], list[int]]


@dataclass(frozen=True)
class Run:
    name: str
    records: dict[str, dict[str, Any]]
    report: dict[str, Any]


def load_run(directory: Path) -> Run:
    lines = (directory / "records.jsonl").read_text().splitlines()
    records = {r["example_id"]: r for r in map(json.loads, lines)}
    report = json.loads((directory / "report.json").read_text())
    return Run(directory.name, records, report)


def load_runs(root: Path) -> list[Run]:
    return [load_run(d) for d in sorted(root.iterdir()) if (d / "records.jsonl").exists()]


def _declined(r: dict[str, Any]) -> bool:
    return bool(r["decision"] == "decline")


METRICS: dict[str, Callable[[dict[str, Any]], tuple[int, int]]] = {
    "hallucinated claim rate": lambda r: (r["shown_hallucinated"], r["shown"]),
    "answer hallucination rate": lambda r: (
        int(r["shown_hallucinated"] > 0 and not _declined(r)),
        int(not _declined(r)),
    ),
    "over-refusal rate": lambda r: (int(_declined(r)), 1),
}


def metric_arrays(run: Run, metric: str, ids: list[str]) -> Pair:
    pairs = [METRICS[metric](run.records[i]) for i in ids]
    return [p[0] for p in pairs], [p[1] for p in pairs]


@dataclass(frozen=True)
class Row:
    name: str
    estimates: dict[str, Estimate]
    tests: dict[str, Comparison]
    """Against the reference run; empty for the reference itself."""
    latency_p50_s: float | None
    cost_per_answer_usd: float | None


def compare(
    runs: list[Run], reference: str = "baseline", *, resamples: int = 10_000, seed: int = 13
) -> tuple[list[Row], int]:
    """Rows for every run, restricted to the responses all runs share."""
    by_name = {r.name: r for r in runs}
    if reference not in by_name:
        raise ValueError(f"no run named {reference!r}")
    shared = sorted(set.intersection(*(set(r.records) for r in runs)))
    ref = by_name[reference]
    rows = []
    for run in [ref, *(r for r in runs if r.name != reference)]:
        estimates, tests = {}, {}
        for metric in METRICS:
            num, den = metric_arrays(run, metric, shared)
            estimates[metric] = bootstrap_ci(num, den, n_resamples=resamples, seed=seed)
            if run is not ref:
                ref_num, ref_den = metric_arrays(ref, metric, shared)
                tests[metric] = paired_permutation_test(
                    ref_num, ref_den, num, den, n_permutations=resamples, seed=seed
                )
        guarded = run.report["config"].get("verifier") is not None
        rows.append(
            Row(
                run.name,
                estimates,
                tests,
                run.report["latency_p50_s"] if guarded else None,
                run.report["cost_per_answer_usd"] if guarded else None,
            )
        )
    return rows, len(shared)


def _pct(e: Estimate) -> str:
    return f"{e.value * 100:.1f}% [{e.low * 100:.1f}, {e.high * 100:.1f}]"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def render(rows: list[Row], n: int, reference: str = "baseline") -> str:
    metrics = list(METRICS)
    lines = [
        "# RAGTruth comparison",
        "",
        f"n = {n} test responses shared by every run. Brackets are 95% bootstrap "
        f"CIs over responses; p-values are two-sided paired permutation tests "
        f"against `{reference}` on the same responses.",
        "",
        "| System | "
        + " | ".join(m.capitalize() for m in metrics)
        + " | Latency p50 | Cost / answer |",
        "|---|" + "---|" * (len(metrics) + 2),
    ]
    for row in rows:
        cells = []
        for m in metrics:
            cell = _pct(row.estimates[m])
            if m in row.tests:
                cell += f" ({_p(row.tests[m].p_value)})"
            cells.append(cell)
        latency = "—" if row.latency_p50_s is None else f"{row.latency_p50_s:.2f} s"
        cost = row.cost_per_answer_usd
        cost_cell = (
            "—" if row.latency_p50_s is None else ("unknown" if cost is None else f"${cost:.4f}")
        )
        lines.append(f"| {row.name} | " + " | ".join(cells) + f" | {latency} | {cost_cell} |")
    return "\n".join(lines) + "\n"


def write_summary(root: Path, reference: str = "baseline") -> Path:
    rows, n = compare(load_runs(root), reference)
    out = root / "summary.md"
    out.write_text(render(rows, n, reference))
    return out
