"""Small Gradio demo (install the `demo` extra): see the guard work claim by claim.

Two tabs. "Check an answer" takes source passages and an answer you paste
in, so it runs with only a verifier. "Ask the docs" runs the full pipeline
over the indexed Kubernetes docs and needs an LLM endpoint. Both have a
guard on/off toggle, so the unguarded answer can be compared with what the
guard lets through.

The view logic lives in plain functions that return strings and rows; only
`build_demo` touches Gradio, so the rest is tested without it installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groundguard.chunking import Chunk
from groundguard.generation import Answer, AnswerGenerator, Claim, StructuredAnswer
from groundguard.guard import Guard, GuardConfig, GuardResult, PolicyMode
from groundguard.llm import LLMClient
from groundguard.ragtruth import sentence_spans
from groundguard.retrieval import HybridRetriever
from groundguard.verifiers import Verifier

CLAIM_HEADERS = ["#", "Claim", "Cites", "Verdict", "Score", "Shown"]


@dataclass(frozen=True)
class View:
    message: str
    """Markdown: what the user would see."""
    claims: list[list[Any]]
    """One row per claim, matching CLAIM_HEADERS."""
    sources: str
    """Markdown listing of the numbered source chunks."""


def parse_sources(text: str) -> list[Chunk]:
    """Passages separated by blank lines become numbered source chunks."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [
        Chunk(f"source-{i}", f"source-{i}", f"Source {i}", 0, len(b), b)
        for i, b in enumerate(blocks, start=1)
    ]


def answer_from_text(text: str, sources: list[Chunk]) -> Answer:
    """Split pasted text into sentence claims that cite every source."""
    every = list(range(1, len(sources) + 1))
    claims = [Claim(text=text[s:e], sources=every) for s, e in sentence_spans(text)]
    return Answer("", sources, StructuredAnswer(answerable=bool(claims), claims=claims), [])


def _refs(cited: list[int]) -> str:
    return "".join(f"[{n}]" for n in cited)


def render_sources(sources: list[Chunk]) -> str:
    if not sources:
        return "_No sources._"
    return "\n\n".join(
        f"**[{i}] {c.section}** `{c.source}`\n\n> {c.text.replace(chr(10), ' ')}"
        for i, c in enumerate(sources, start=1)
    )


def unguarded_view(answer: Answer) -> View:
    claims = answer.claims
    message = (
        " ".join(f"{c.text} {_refs(c.sources)}".strip() for c in claims)
        if claims
        else "_The model said it can't answer from the sources._"
    )
    rows = [
        [i, c.text, _refs(c.sources), "not checked", "", "yes"] for i, c in enumerate(claims, 1)
    ]
    return View(message, rows, render_sources(answer.sources))


def guarded_view(answer: Answer, result: GuardResult) -> View:
    kept = {id(r) for r in result.kept}
    rows = [
        [
            i,
            r.claim.text,
            _refs(r.claim.cited),
            str(r.verification.verdict) if r.claim.citations_ok else str(r.claim.status),
            round(r.verification.score, 3),
            "yes" if id(r) in kept else "no",
        ]
        for i, r in enumerate(result.claims, start=1)
    ]
    header = f"**Decision: {result.decision}** · {result.support_ratio:.0%} of claims supported"
    return View(f"{header}\n\n{result.message}", rows, render_sources(answer.sources))


def policy(mode: str, min_support: float) -> GuardConfig:
    return GuardConfig(mode=PolicyMode(mode), min_support=min_support)


def check_answer(
    verifier: Verifier,
    sources_text: str,
    answer_text: str,
    guard_on: bool,
    mode: str = "drop",
    min_support: float = 0.5,
) -> View:
    sources = parse_sources(sources_text)
    if not sources or not answer_text.strip():
        return View("_Paste at least one source passage and an answer._", [], "")
    answer = answer_from_text(answer_text, sources)
    if not guard_on:
        return unguarded_view(answer)
    return guarded_view(answer, Guard(verifier, policy(mode, min_support)).check(answer))


def ask(
    verifier: Verifier,
    retriever: HybridRetriever,
    client: LLMClient,
    question: str,
    guard_on: bool,
    mode: str = "drop",
    min_support: float = 0.5,
    k: int = 6,
) -> View:
    if not question.strip():
        return View("_Type a question._", [], "")
    chunks = [h.chunk for h in retriever.search(question, k=k)]
    answer = AnswerGenerator(client).generate(question, chunks)
    if not guard_on:
        return unguarded_view(answer)
    return guarded_view(answer, Guard(verifier, policy(mode, min_support)).check(answer))


EXAMPLE_SOURCES = (
    "A Job creates one or more Pods and will continue to retry execution of "
    "the Pods until a specified number of them successfully terminate.\n\n"
    "The backoffLimit field sets the number of retries before a Job is "
    "considered failed. The default is 6."
)
EXAMPLE_ANSWER = (
    "A Job retries its Pods until enough of them finish successfully. "
    "By default it gives up after 6 retries. Failed Jobs are deleted after one hour."
)


def build_demo(
    verifier: Verifier,
    retriever: HybridRetriever | None = None,
    client: LLMClient | None = None,
) -> Any:
    import gradio as gr

    def as_outputs(view: View) -> tuple[str, list[list[Any]], str]:
        return view.message, view.claims, view.sources

    with gr.Blocks(title="groundguard") as demo:
        gr.Markdown(
            "# groundguard\nEvery claim is checked against its sources; "
            "unsupported claims are dropped, and the answer is declined when too "
            "little is left. Toggle the guard to compare."
        )
        with gr.Row():
            guard_on = gr.Checkbox(value=True, label="Guard on")
            mode = gr.Radio(["drop", "strict"], value="drop", label="Policy")
            min_support = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Min support")

        with gr.Tab("Check an answer"):
            sources_in = gr.Textbox(
                EXAMPLE_SOURCES, lines=6, label="Sources (blank line between passages)"
            )
            answer_in = gr.Textbox(EXAMPLE_ANSWER, lines=3, label="Answer to check")
            check_btn = gr.Button("Check", variant="primary")

        with gr.Tab("Ask the docs"):
            if retriever is None or client is None:
                gr.Markdown(
                    "_Needs an index (`groundguard ingest`) and `LLM_BASE_URL` / "
                    "`LLM_MODEL`; not available in this instance._"
                )
            question_in = gr.Textbox(
                label="Question", placeholder="How many times does a Job retry?"
            )
            ask_btn = gr.Button("Ask", variant="primary", interactive=retriever is not None)

        message_out = gr.Markdown()
        claims_out = gr.Dataframe(headers=CLAIM_HEADERS, label="Claims", wrap=True)
        with gr.Accordion("Sources", open=False):
            sources_out = gr.Markdown()
        outputs = [message_out, claims_out, sources_out]

        check_btn.click(
            lambda s, a, g, m, t: as_outputs(check_answer(verifier, s, a, g, m, t)),
            [sources_in, answer_in, guard_on, mode, min_support],
            outputs,
        )
        if retriever is not None and client is not None:
            ask_btn.click(
                lambda q, g, m, t: as_outputs(ask(verifier, retriever, client, q, g, m, t)),
                [question_in, guard_on, mode, min_support],
                outputs,
            )
    return demo
