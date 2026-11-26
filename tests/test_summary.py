import json
from pathlib import Path

import pytest

from groundguard import cli
from groundguard.summary import compare, load_runs, render, write_summary


def record(i: int, decision: str, shown: int, bad: int) -> dict[str, object]:
    return {
        "example_id": str(i),
        "task_type": "QA",
        "decision": decision,
        "claims": 2,
        "shown": shown,
        "shown_hallucinated": bad,
        "gold_hallucinated": 1,
        "flagged": 0,
        "caught": 0,
        "latency_s": 0.5,
        "cost_usd": 0.0,
    }


def write_run(root: Path, name: str, records: list[dict[str, object]], guarded: bool) -> None:
    out = root / name
    out.mkdir(parents=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    report = {
        "config": {"verifier": {"kind": "nli"} if guarded else None},
        "latency_p50_s": 0.5,
        "cost_per_answer_usd": 0.0,
    }
    (out / "report.json").write_text(json.dumps(report))


@pytest.fixture
def results(tmp_path: Path) -> Path:
    write_run(tmp_path, "baseline", [record(i, "answer", 2, 1) for i in range(20)], False)
    guarded = [record(i, "partial", 1, 0) for i in range(18)]
    guarded += [record(18, "decline", 0, 0), record(19, "answer", 2, 1)]
    write_run(tmp_path, "nli", guarded, True)
    (tmp_path / "unfinished").mkdir()  # no records yet: skipped
    return tmp_path


def test_reference_comes_first_and_is_not_tested(results: Path) -> None:
    rows, n = compare(load_runs(results), resamples=500)
    assert n == 20
    assert [r.name for r in rows] == ["baseline", "nli"]
    assert rows[0].tests == {}
    assert rows[0].latency_p50_s is None


def test_rates_and_paired_tests(results: Path) -> None:
    rows, _ = compare(load_runs(results), resamples=500)
    nli = rows[1]
    assert nli.estimates["hallucinated claim rate"].value == pytest.approx(1 / 20)
    assert nli.estimates["over-refusal rate"].value == pytest.approx(1 / 20)
    assert nli.estimates["answer hallucination rate"].value == pytest.approx(1 / 19)
    assert nli.tests["hallucinated claim rate"].p_value < 0.01
    # One decline out of 20 is not distinguishable from zero.
    assert nli.tests["over-refusal rate"].p_value > 0.5


def test_only_shared_responses_are_compared(results: Path) -> None:
    extra = results / "nli" / "records.jsonl"
    extra.write_text(extra.read_text() + json.dumps(record(99, "answer", 2, 2)) + "\n")
    rows, n = compare(load_runs(results), resamples=200)
    assert n == 20
    assert rows[1].estimates["hallucinated claim rate"].value == pytest.approx(1 / 20)


def test_unknown_reference_is_an_error(results: Path) -> None:
    with pytest.raises(ValueError):
        compare(load_runs(results), reference="judge")


def test_markdown_and_cli(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows, n = compare(load_runs(results), resamples=200)
    text = render(rows, n)
    assert "| baseline | 50.0% [50.0, 50.0] |" in text
    assert "(p = " in text or "(p < 0.001)" in text

    assert cli.main(["compare", str(results)]) == 0
    assert (results / "summary.md").read_text() == capsys.readouterr().out.rstrip("\n") + "\n"
    assert write_summary(results).name == "summary.md"
