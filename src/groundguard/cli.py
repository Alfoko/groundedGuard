"""Command line entry point: `groundguard ingest` and `groundguard ask`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from groundguard import corpus
from groundguard.generation import Answer, AnswerGenerator
from groundguard.llm import LLMClient, LLMError, OpenAICompatibleClient
from groundguard.retrieval import Embedder

DEFAULT_INDEX = Path("data/index/k8s")
DEFAULT_CHECKOUT = Path("data/raw/k8s-website")


def _embedder(model: str | None) -> Embedder | None:
    if not model:
        return None
    from groundguard.retrieval import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model)


def cmd_ingest(args: argparse.Namespace) -> int:
    if args.docs:
        docs_root = Path(args.docs)
        revision = "local"
    else:
        print(f"Fetching kubernetes/website@{args.commit[:12]} ...", file=sys.stderr)
        docs_root = corpus.fetch_kubernetes_docs(Path(args.checkout), args.commit)
        revision = args.commit
    docs = corpus.load_hugo_docs(docs_root)
    chunks = corpus.build_index(
        docs,
        Path(args.index),
        max_chars=args.max_chars,
        overlap=args.overlap,
        embedder=_embedder(args.embed),
        extra_meta={"corpus": "kubernetes/website", "revision": revision},
    )
    print(f"Indexed {len(docs)} documents as {len(chunks)} chunks in {args.index}")
    return 0


def format_answer(answer: Answer) -> str:
    if not answer.claims:
        return "I can't answer that from the documents."
    lines = []
    for claim in answer.claims:
        refs = "".join(f"[{n}]" for n in claim.sources)
        lines.append(f"- {claim.text} {refs}".rstrip())
    cited = sorted({n for c in answer.claims for n in c.sources})
    lines.append("")
    lines.append("Sources:")
    for n in cited:
        chunk = answer.chunk_for(n)
        label = f"{chunk.section} ({chunk.source})" if chunk else "(not a provided source)"
        lines.append(f"  [{n}] {label}")
    return "\n".join(lines)


def cmd_ask(args: argparse.Namespace, client: LLMClient | None = None) -> int:
    index_dir = Path(args.index)
    if not (index_dir / "chunks.jsonl").exists():
        print(f"No index at {index_dir}; run `groundguard ingest` first.", file=sys.stderr)
        return 2
    if client is None:
        try:
            client = OpenAICompatibleClient.from_env()
        except LLMError as exc:
            print(f"LLM not configured: {exc}", file=sys.stderr)
            return 2
    retriever = corpus.load_retriever(index_dir, _embedder(args.embed))
    hits = retriever.search(args.question, k=args.k)
    answer = AnswerGenerator(client).generate(args.question, [h.chunk for h in hits])
    if args.json:
        print(
            json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.structured.model_dump(),
                    "sources": [h.chunk.id for h in hits],
                    "cost_usd": answer.usage.cost_usd,
                },
                indent=2,
            )
        )
    else:
        print(format_answer(answer))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from groundguard.evaluation import EvalConfig, run_eval, write_report

    config = EvalConfig.from_yaml(Path(args.config))
    run = run_eval(config)
    out = write_report(run)
    print((out / "report.md").read_text())
    if run.stopped_early:
        print(f"Stopped early: {run.stopped_early}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from groundguard.service import app_from_env

    uvicorn.run(app_from_env(args.index, args.verifier), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groundguard")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download the Kubernetes docs and build an index")
    ingest.add_argument("--index", default=str(DEFAULT_INDEX))
    ingest.add_argument("--checkout", default=str(DEFAULT_CHECKOUT))
    ingest.add_argument("--commit", default=corpus.K8S_COMMIT)
    ingest.add_argument("--docs", help="index a local Hugo docs tree instead of fetching")
    ingest.add_argument("--max-chars", type=int, default=1200)
    ingest.add_argument("--overlap", type=int, default=150)
    ingest.add_argument("--embed", metavar="MODEL", help="also store dense vectors")

    ask = sub.add_parser("ask", help="answer a question from the indexed docs")
    ask.add_argument("question")
    ask.add_argument("--index", default=str(DEFAULT_INDEX))
    ask.add_argument("-k", type=int, default=6, help="number of chunks to retrieve")
    ask.add_argument("--embed", metavar="MODEL", help="embedding model used at ingest")
    ask.add_argument("--json", action="store_true")

    ev = sub.add_parser("eval", help="run an evaluation config on RAGTruth")
    ev.add_argument("config", help="path to a YAML config in configs/")

    serve = sub.add_parser("serve", help="run the HTTP service (needs the service extra)")
    serve.add_argument("--index", default=str(DEFAULT_INDEX))
    serve.add_argument("--verifier", choices=["nli", "llm_judge"], default="nli")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "serve":
        return cmd_serve(args)
    return cmd_ask(args)


if __name__ == "__main__":
    raise SystemExit(main())
