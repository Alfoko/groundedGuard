# Results

Every number on this page comes from committed run output under
[`results/ragtruth/`](../results/ragtruth/). Each run directory holds the
config that produced it (inside `report.json`), the per-response records and
the rendered report. The comparison table is
[`results/ragtruth/summary.md`](../results/ragtruth/summary.md), written by
`groundguard compare`.

## Headline

On a seeded sample of 300 RAGTruth test responses, putting the guard with
the NLI verifier in front of the responses:

| System | Hallucinated claim rate | Answer hallucination rate | Over-refusal rate | Latency p50 | Cost / answer |
|---|---|---|---|---|---|
| No guard | 8.3% [6.0, 11.0] | 32.0% [26.7, 37.3] | 0.0% | — | — |
| Guard + NLI | 2.3% [1.5, 3.2] | 11.1% [7.3, 15.1] | 15.7% [11.7, 19.7] | 4.46 s | $0 (local) |
| Guard + LLM judge | TBD | TBD | TBD | TBD | TBD |
| Guard + LLM judge, strict | TBD | TBD | TBD | TBD | TBD |
| LLM self-check | TBD | TBD | TBD | TBD | TBD |

Brackets are 95% bootstrap intervals over responses. All three differences
between the NLI guard and no guard have paired permutation p < 0.001.

NLI verifier, claim level: precision 19.9% [14.3, 26.2], recall 76.8%
[68.4, 83.0] for flagging gold-hallucinated claims. Latency p95 is 16.9 s.

The LLM judge and self-check rows are blank because their runs have not
happened: the API account used for the project has no credit. The configs
are in [`configs/ragtruth/`](../configs/ragtruth/) and run with
`groundguard eval configs/ragtruth/judge.yaml` (and so on) once an
OpenAI-compatible endpoint is set in `LLM_BASE_URL` / `LLM_MODEL`.

## What the numbers say

The NLI guard removes most hallucinated content from what users would see:
the share of shown claims that RAGTruth annotators marked as hallucinated
drops from 8.3% to 2.3%, and the share of answered responses containing at
least one drops from 32.0% to 11.1%.

It pays for that with a lot of false alarms. Only about one in five claims
it flags is actually labelled hallucinated, so most dropped claims were
fine. That shows up as 15.7% of responses declined outright, and many more
answered with some correct claims removed, even though every RAGTruth
question is answerable from its context. An off-the-shelf MNLI model is
strict about paraphrase, summarisation and claims that need two passages
together, and the threshold was not tuned (see limitations).

So on this sample the NLI guard is a high-recall, low-precision filter.
Whether that trade is acceptable depends on the application; the LLM judge
runs are the obvious next comparison, because a judge sees all passages at
once and should reject far fewer correct summaries.

## Methodology

**Data.** RAGTruth (commit `c103204` of ParticleMedia/RAGTruth), official
split. A dev slice of 10% of train is held out by hashing `source_id`
(seed 13), so no source context appears in more than one of train, dev and
test. Reported runs use a seeded random sample of 300 test responses (seed
13), drawn across all three task types (QA, Summary, Data2txt). The same
300 responses are used for every system, so comparisons are paired.

**Claims and labels.** RAGTruth responses come with character-level spans
marking hallucinated text, not claims. Each response is split into
sentences with a deterministic rule-based splitter, and a sentence counts as
hallucinated if it overlaps any gold span. Sentence-level labels are coarser
than the annotation: a sentence with one wrong number counts as fully
hallucinated.

**Guard setup.** RAGTruth responses carry no citations, so each sentence is
treated as a claim citing every chunk of its source context (context split
into 1200-character chunks with 150 characters of overlap). The verifier
decides per claim; the policy then keeps supported claims, drops the rest,
and declines when fewer than half of a response's claims are supported
(`mode: drop`, `min_support: 0.5`).

**NLI verifier.** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, fp32 on
CPU. Each (chunk, claim) pair is scored separately and the best-supporting
chunk decides; a claim is supported when its entailment probability is at
least 0.5.

**Metrics.** All rates are ratios of sums over responses:

- hallucinated claim rate: gold-hallucinated shown claims / shown claims
- answer hallucination rate: answered responses with any gold-hallucinated
  shown claim / answered responses
- over-refusal rate: declined responses / responses (all are answerable)
- verifier precision and recall: flagged claims that are gold-hallucinated,
  and gold-hallucinated claims that were flagged

Intervals are percentile bootstrap over responses (10,000 resamples), since
claims within one response are correlated. Differences use a two-sided
paired permutation test that swaps system labels per response.

**Latency and cost.** Latency is the guard's wall time per response on a
4-core CPU container with no GPU. NLI runs locally, so it has no API cost.

## Limitations

- **Only two systems measured.** The LLM judge and self-check baselines are
  implemented and configured but not run, for lack of API credit.
- **No threshold tuning.** The NLI threshold (0.5) and `min_support` (0.5)
  are defaults. The dev slice exists so they can be tuned without touching
  test, but that tuning has not been done, so the precision/recall balance
  above is not the best this verifier can reach.
- **Not the guard's intended input.** The guard is designed for answers
  whose claims cite specific chunks. RAGTruth responses have no citations,
  so every claim is checked against the whole context. Per-claim citations
  would make checks cheaper and stricter.
- **Sentence-level labels** overstate how much of a sentence is wrong and
  give no credit for dropping a sentence that is mostly right.
- **Sample size.** At 300 responses the intervals above are several points
  wide on the answer-level rates, and per-task breakdowns would be noisier
  still, so none are reported.
- **Latency is CPU-only**, and one response in the NLI run (of 300) was timed
  while two worker threads were contending for the CPU, before the config
  was changed to one worker; its latency is inflated. It is one value of
  300, so the median is unaffected; the p95 includes it.
- **One generator family.** RAGTruth responses come from six models of 2023;
  how well the numbers transfer to current models is untested.
