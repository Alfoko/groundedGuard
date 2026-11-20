import json
from pathlib import Path

import pytest

from groundguard import cli, corpus
from groundguard.llm import FakeLLMClient

POD_PAGE = """---
title: Pods
content_type: concept
---

<!-- overview -->

_Pods_ are the smallest deployable units. A Pod wraps one or more
{{< glossary_tooltip text="containers" term_id="container" >}}.

{{< note >}}
You need a container runtime on each node.
{{< /note >}}

## Restart policy

The restartPolicy field applies to all containers in the Pod.
"""

CRON_PAGE = """---
title: CronJob
---

A CronJob creates Jobs on a repeating schedule written in cron format.
"""


@pytest.fixture
def docs_tree(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "concepts" / "pods").mkdir(parents=True)
    (root / "concepts" / "pods" / "_index.md").write_text(POD_PAGE)
    (root / "concepts" / "cronjob.md").write_text(CRON_PAGE)
    (root / "empty.md").write_text("---\ntitle: Empty\n---\n<!-- nothing -->\n")
    return root


def test_clean_hugo_keeps_tooltip_text_and_drops_markup() -> None:
    cleaned = corpus.clean_hugo(POD_PAGE)
    assert "one or more\ncontainers." in cleaned
    assert "{{" not in cleaned
    assert "<!--" not in cleaned
    assert "You need a container runtime" in cleaned


def test_load_hugo_docs_skips_empty_pages(docs_tree: Path) -> None:
    docs = corpus.load_hugo_docs(docs_tree)
    assert [d.source for d in docs] == ["concepts/cronjob.md", "concepts/pods/_index.md"]
    assert docs[1].title == "Pods"


def test_index_round_trips_through_disk(docs_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "index"
    chunks = corpus.build_index(
        corpus.load_hugo_docs(docs_tree),
        out,
        max_chars=200,
        overlap=20,
        extra_meta={"revision": "test"},
    )
    assert corpus.load_chunks(out) == chunks
    meta = json.loads((out / "meta.json").read_text())
    assert meta["num_chunks"] == len(chunks)
    assert meta["revision"] == "test"
    assert not (out / "vectors.npy").exists()

    hits = corpus.load_retriever(out).search("restart policy containers", k=1)
    assert hits[0].chunk.section == "Pods > Restart policy"


def test_ingest_from_local_docs(
    docs_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "index"
    assert cli.main(["ingest", "--docs", str(docs_tree), "--index", str(out)]) == 0
    assert "Indexed 2 documents" in capsys.readouterr().out
    assert json.loads((out / "meta.json").read_text())["revision"] == "local"


def test_ask_prints_claims_with_numbered_sources(
    docs_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "index"
    cli.main(["ingest", "--docs", str(docs_tree), "--index", str(out)])
    capsys.readouterr()

    reply = json.dumps(
        {
            "answerable": True,
            "claims": [{"text": "CronJobs run Jobs on a schedule.", "sources": [1]}],
        }
    )
    fake = FakeLLMClient([reply])
    args = cli.build_parser().parse_args(["ask", "what does a cronjob do", "--index", str(out)])
    assert cli.cmd_ask(args, client=fake) == 0

    printed = capsys.readouterr().out
    assert "- CronJobs run Jobs on a schedule. [1]" in printed
    assert "[1] CronJob (concepts/cronjob.md)" in printed
    # The retrieved chunks were what the model saw.
    assert "repeating schedule" in fake.calls[0][1]["content"]


def test_ask_without_index_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["ask", "anything", "--index", str(tmp_path / "missing")]) == 2
    assert "groundguard ingest" in capsys.readouterr().err


def test_unanswerable_answer_declines() -> None:
    from groundguard.generation import Answer, StructuredAnswer

    answer = Answer("q", [], StructuredAnswer(answerable=False), [])
    assert cli.format_answer(answer) == "I can't answer that from the documents."
