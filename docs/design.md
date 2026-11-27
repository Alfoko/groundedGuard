# Design

A short tour of how groundguard is put together and why. The README covers
what it does and how well; this covers the decisions behind it.

## Data flow

```
question
  -> HybridRetriever.search         retrieval.py   BM25 (+ dense), fused by RRF
  -> AnswerGenerator.generate       generation.py  numbered sources in, JSON claims out
  -> check_citations                citations.py   cited numbers must exist
  -> Verifier.verify (per claim)    verifiers.py   NLI / LLM judge / self-check
  -> apply_policy                   guard.py       answer / partial / decline
```

`Guard.check` runs the last three steps on any `Answer`. Everything before it
is optional: `/verify`, the demo's first tab and the RAGTruth eval all build
an `Answer` from someone else's text and skip retrieval and generation.

## Decisions

**Claims, not answers, are the unit.** One unsupported sentence in an
otherwise good answer is the failure that matters most and the one
answer-level scores hide. The generator is asked for claims directly, as
JSON validated with pydantic, so the guard never has to re-split prose; when
the input is prose (RAGTruth, `/verify` with `answer`), a deterministic
sentence splitter stands in.

**Citations are checked before anything costs money.** A claim with no
citation, or one citing only source numbers the model wasn't shown, can't be
grounded, so it fails in `check_citations` without a verifier call. The
generator retries once with the validation error as feedback when the JSON is
malformed.

**Verifiers share one small protocol.** `verify(claim, evidence) ->
Verification` with a verdict (supported, contradicted, not enough info), a
score and the deciding chunk. The guard, eval harness, service and demo only
see the protocol, which is also what lets every test run with a stub.
Verifiers that work better on a whole answer at once (the self-check
baseline) add `verify_batch`, and the guard uses it when present.

**Fail closed.** An LLM judge reply that doesn't parse is "not enough
info", never "supported". For a guard, a false refusal is recoverable and a
false pass is the thing it exists to prevent.

**NLI scores chunks one at a time.** Each cited chunk is scored as a separate
premise and the best-supporting one wins. That keeps every premise inside
the model's 512-token window, at the cost of rejecting claims that only
follow from two chunks together. The LLM judge sees all cited passages at
once and doesn't have this limit.

**Two policies and a floor.** `drop` removes unsupported claims and keeps
the rest; `strict` rejects the answer if any claim fails. Both decline when
the supported share falls below `min_support`. A decline names up to three
claims that couldn't be confirmed and why, so a user learns what the
documents don't say rather than getting a bare refusal.

**Retrieval fuses ranks, not scores.** BM25 and cosine similarities live on
unrelated scales, so reciprocal rank fusion merges the two rankings by rank
alone. The dense half is optional; without an embedding model the retriever
is BM25 only. Chunks keep their source file, section path and character
offsets so every citation can be traced to a place in the docs.

## Evaluation

**RAGTruth as the benchmark.** It has human span-level hallucination labels
on responses from several models over QA, summarisation and data-to-text
tasks, which is what a grounding guard needs to be scored against. Spans are
mapped to sentence claims by overlap.

**Splits without leakage.** The official test split is only used for
reported runs. Tuning is meant for a dev slice of train, held out by hashing
the source context id, so no context appears on both sides.

**Ratio-of-sums metrics, resampled by answer.** Every rate is
sum(numerator) / sum(denominator) over responses. Bootstrap intervals and
the paired permutation test resample or swap whole responses, because claims
within a response are correlated and treating them as independent would
make intervals too narrow. All systems run on the same seeded sample, so
comparisons are paired.

**Runs are reproducible and cheap to repeat.** A run is a YAML config plus a
seed. LLM calls go through a SQLite cache keyed by the full request, a
`budget_usd` cap stops a run cleanly when reached, and each finished response
is appended to a progress file so an interrupted run resumes. The progress
file carries a fingerprint of the result-affecting config fields, so changing
the verifier or policy starts afresh while changing the worker count doesn't.

**Reports come from records.** `groundguard eval` writes per-response
records and a report; `groundguard compare` builds the cross-system table
from the records alone. The numbers in the README and docs are copied from
those files.

## Packaging

The core depends only on httpx, numpy, pydantic and PyYAML. torch,
transformers, sentence-transformers, FastAPI and Gradio are optional extras,
and their imports are deferred to the functions that need them, so the core
tests run without any of them and never touch the network. The Docker image
installs CPU-only torch from the PyTorch index so it doesn't pull CUDA.
