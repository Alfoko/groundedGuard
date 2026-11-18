import itertools
from pathlib import Path

import pytest

from groundguard.chunking import chunk_document
from groundguard.documents import iter_documents, parse_markdown

POD_DOC = """---
title: "Pods"
weight: 10
---

Pods are the smallest deployable units of computing.

## Using Pods

A Pod is usually created by a workload resource.

### Restart policy

The restartPolicy applies to all containers in the Pod.

```yaml
# not a heading
kind: Pod
```

## Pod lifecycle

Pods follow a defined lifecycle.
"""


def test_front_matter_title_and_body_offset() -> None:
    doc = parse_markdown("concepts/pods.md", POD_DOC)
    assert doc.title == "Pods"
    assert doc.text[doc.body_start :].lstrip().startswith("Pods are")


def test_title_falls_back_to_file_name() -> None:
    doc = parse_markdown("tasks/run-a-job.md", "Some text.\n")
    assert doc.title == "run a job"
    assert doc.body_start == 0


def test_sections_follow_heading_hierarchy() -> None:
    chunks = chunk_document(parse_markdown("concepts/pods.md", POD_DOC))
    assert [c.section for c in chunks] == [
        "Pods",
        "Pods > Using Pods",
        "Pods > Using Pods > Restart policy",
        "Pods > Pod lifecycle",
    ]


def test_heading_inside_code_fence_is_ignored() -> None:
    chunks = chunk_document(parse_markdown("concepts/pods.md", POD_DOC))
    restart = next(c for c in chunks if c.section.endswith("Restart policy"))
    assert "# not a heading" in restart.text


def test_offsets_point_back_into_source() -> None:
    doc = parse_markdown("concepts/pods.md", POD_DOC)
    for chunk in chunk_document(doc):
        assert doc.text[chunk.start : chunk.end] == chunk.text
        assert chunk.text == chunk.text.strip()


def test_ids_are_unique_and_stable() -> None:
    doc = parse_markdown("concepts/pods.md", POD_DOC)
    first = [c.id for c in chunk_document(doc)]
    assert len(set(first)) == len(first)
    assert first == [c.id for c in chunk_document(doc)]
    assert first[0] == "concepts/pods.md#0"


def test_long_section_is_windowed_with_overlap() -> None:
    sentences = " ".join(f"Sentence number {i} talks about nodes." for i in range(60))
    doc = parse_markdown("long.md", f"# Long\n\n{sentences}\n")
    chunks = chunk_document(doc, max_chars=300, overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start < prev.end, "consecutive windows should overlap"
        assert nxt.start > prev.start
    # Windows prefer sentence boundaries over cutting mid-sentence.
    assert all(c.text.endswith(".") for c in chunks)
    # Together the windows cover the whole section, heading line included.
    assert chunks[0].start == 0
    assert chunks[0].text.startswith("# Long")
    assert chunks[-1].end == len(doc.text.rstrip())


def test_unbreakable_text_is_cut_hard() -> None:
    doc = parse_markdown("blob.md", "x" * 1000)
    chunks = chunk_document(doc, max_chars=300, overlap=0)
    assert [len(c.text) for c in chunks] == [300, 300, 300, 100]


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_rejects_bad_window_settings(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(parse_markdown("a.md", "text"), max_chars=max_chars, overlap=overlap)


def test_iter_documents_reads_tree_in_order(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.md").write_text("---\ntitle: Two\n---\nBody two.\n")
    (tmp_path / "a.md").write_text("Body one.\n")
    (tmp_path / "notes.txt").write_text("ignored\n")

    docs = list(iter_documents(tmp_path))
    assert [d.source for d in docs] == ["a.md", "b/two.md"]
    assert docs[1].title == "Two"
