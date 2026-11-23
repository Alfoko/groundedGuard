import json
from pathlib import Path

import pytest

from groundguard.ragtruth import (
    Example,
    Span,
    dev_bucket,
    gold_claims,
    label_claims,
    load_ragtruth,
    select,
    sentence_spans,
)


def write_dataset(root: Path, n_sources: int = 40) -> None:
    (root / "dataset").mkdir(parents=True)
    sources, responses = [], []
    for i in range(n_sources):
        task = ["QA", "Summary", "Data2txt"][i % 3]
        info: object
        if task == "QA":
            info = {"question": f"question {i}", "passages": f"passage 1: fact {i}\n\n"}
        elif task == "Summary":
            info = f"Article {i} text."
        else:
            info = {"name": f"Shop {i}", "city": "Sydney"}
        sources.append({"source_id": str(i), "task_type": task, "source_info": info})
        for model in ("m1", "m2"):
            responses.append(
                {
                    "id": f"{i}-{model}",
                    "source_id": str(i),
                    "model": model,
                    "split": "test" if i >= 30 else "train",
                    "quality": "good",
                    "response": "It is open. It closes at 9pm.",
                    "labels": [
                        {
                            "start": 12,
                            "end": 29,
                            "text": "It closes at 9pm.",
                            "label_type": "Evident Baseless Info",
                        }
                    ]
                    if model == "m2"
                    else [],
                }
            )
    for name, rows in (("source_info", sources), ("response", responses)):
        with (root / "dataset" / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


def test_loader_builds_contexts_per_task(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    examples = {e.id: e for e in load_ragtruth(tmp_path)}
    qa, summary, d2t = examples["0-m1"], examples["1-m1"], examples["2-m1"]
    assert qa.question == "question 0"
    assert qa.context == "passage 1: fact 0"
    assert summary.question == ""
    assert summary.context == "Article 1 text."
    assert json.loads(d2t.context) == {"name": "Shop 2", "city": "Sydney"}
    assert examples["0-m2"].spans == [Span(12, 29, "Evident Baseless Info")]
    assert examples["0-m2"].hallucinated and not qa.hallucinated


def test_dev_split_comes_from_train_and_keeps_sources_together(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    examples = load_ragtruth(tmp_path, dev_fraction=0.3, seed=1)
    dev, test = select(examples, "dev"), select(examples, "test")
    assert dev, "expected some dev examples at 30%"
    assert all(int(e.source_id) < 30 for e in dev)
    assert {e.source_id for e in test} == {str(i) for i in range(30, 40)}
    by_source: dict[str, set[str]] = {}
    for e in examples:
        by_source.setdefault(e.source_id, set()).add(e.split)
    assert all(len(splits) == 1 for splits in by_source.values())


def test_dev_assignment_is_deterministic() -> None:
    assert dev_bucket("123", 0.1, 13) == dev_bucket("123", 0.1, 13)
    assert sum(dev_bucket(str(i), 0.1, 13) for i in range(5000)) == pytest.approx(500, abs=75)


def test_sentence_spans_respect_abbreviations_and_newlines() -> None:
    text = (
        "Dr. Smith moved to the U.S. in 1990. He works at Acme Inc. now!\n- Opens 9am\n- Closes 5pm"
    )
    sentences = [text[s:e] for s, e in sentence_spans(text)]
    assert sentences == [
        "Dr. Smith moved to the U.S. in 1990.",
        "He works at Acme Inc. now!",
        "- Opens 9am",
        "- Closes 5pm",
    ]


def test_sentence_spans_handle_quotes_and_initials() -> None:
    text = 'He said "it works." J. K. Rowling agreed.'
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        'He said "it works."',
        "J. K. Rowling agreed.",
    ]


def test_claims_overlapping_a_span_are_hallucinated() -> None:
    gold = [Span(5, 8, "Evident Conflict"), Span(20, 25, "Subtle Baseless Info")]
    assert label_claims([(0, 5), (0, 6), (10, 20), (19, 30)], gold) == [
        [],
        ["Evident Conflict"],
        [],
        ["Subtle Baseless Info"],
    ]


def test_gold_claims_for_an_example() -> None:
    example = Example(
        id="1",
        source_id="1",
        task_type="Summary",
        model="m",
        split="test",
        quality="good",
        question="",
        context="",
        response="It is open. It closes at 9pm.",
        spans=[Span(12, 29, "Evident Baseless Info")],
    )
    claims = gold_claims(example)
    assert [(c.text, c.hallucinated) for c in claims] == [
        ("It is open.", False),
        ("It closes at 9pm.", True),
    ]
    assert claims[1].label_types == ["Evident Baseless Info"]
