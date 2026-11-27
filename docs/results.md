# Results

Every number on this page comes from committed run output under
[`results/ragtruth/`](../results/ragtruth/). Each run directory holds the
config that produced it (inside `report.json`), the per-response records and
the rendered report. The comparison table is
[`results/ragtruth/summary.md`](../results/ragtruth/summary.md), written by
`groundguard compare`.

## Headline

Seeded sample of 300 RAGTruth test responses, the same responses for every
system:

| System | Hallucinated claim rate | Answer hallucination rate | Over-refusal rate | Verifier precision / recall | Latency p50 | Cost / answer |
|---|---|---|---|---|---|---|
| No guard | 8.3% [6.0, 11.0] | 32.0% [26.7, 37.3] | 0.0% | — | — | — |
| Guard + NLI | 2.3% [1.5, 3.2] | 11.1% [7.3, 15.1] | 15.7% [11.7, 19.7] | 19.9% / 76.8% | 4.46 s | $0 (local) |
| Guard + LLM judge | 2.6% [1.7, 3.4] | 13.3% [9.4, 17.4] | 9.7% [6.3, 13.0] | 28.0% / 74.6% | 8.35 s | $0.0011 |
| Guard + LLM judge, strict | 1.4% [0.7, 2.2] | 8.2% [4.1, 12.9] | 51.3% [45.7, 57.0] | 28.0% / 74.6% | see below | $0.0011 |
| LLM self-check (one call) | 5.0% [2.3, 8.3] | 10.5% [7.1, 14.2] | 5.0% [2.7, 7.7] | 18.1% / 54.6% | 0.72 s | $0.00014 |

Brackets are 95% bootstrap intervals over responses. Every guarded system
differs from no guard on all three rates with paired permutation p < 0.001
([`summary.md`](../results/ragtruth/summary.md)). The LLM rows use
`openai/gpt-4o-mini` through OpenRouter. The judge run's calls cost $0.33
and the self-check run's $0.04 (`total_cost_usd` in their reports); the
strict run reused the judge's cached calls and cost nothing extra.

Verifier precision is the share of flagged claims that annotators marked as
hallucinated; recall is the share of hallucinated claims that were flagged.
Their intervals are in each run's `report.md`.

Paired tests against the NLI guard instead of no guard
(`groundguard compare --reference nli`):

| vs. NLI | Hallucinated claim rate | Answer hallucination rate | Over-refusal rate |
|---|---|---|---|
| LLM judge | p = 0.551 | p = 0.301 | p = 0.007 |
| LLM judge, strict | p = 0.093 | p = 0.286 | p < 0.001 |
| LLM self-check | p = 0.110 | p = 0.792 | p < 0.001 |

## What the numbers say

**Every guard helps a lot.** The share of shown claims that RAGTruth
annotators marked as hallucinated falls from 8.3% to between 1.4% and 5.0%,
and the share of answers containing one from 32.0% to between 8.2% and 13.3%.

**The LLM judge matches NLI on hallucinations and refuses less.** On this
sample the judge and the NLI guard can't be told apart on either
hallucination rate (p = 0.55 and 0.30), but the judge declines 9.7% of
answers against NLI's 15.7% (p = 0.007), and fewer of the claims it flags
are false alarms (precision 28.0% against 19.9%). It sees all of a
response's context at once, so it accepts claims NLI rejects because the
support is split across chunks or paraphrased. It costs about a tenth of a
cent per answer and is the slowest option, at a median of 8.4 s per answer
with eight requests in flight.

**Strict mode is too strict.** Rejecting an answer over any single
unsupported claim gives the lowest hallucination rates but declines just
over half of all answers, which no assistant could ship. With verifiers
this imprecise, dropping claims is the only usable policy.

**One-call self-check is cheap and gentle, and misses more.** Reviewing the
whole answer in one call costs about an eighth of the per-claim judge and
declines the fewest answers (5.0%), but it flags only 54.6% of hallucinated
claims, and its hallucinated-claim rate (5.0%) is about twice NLI's,
although with n = 300 that gap is not significant (p = 0.11).

**All the verifiers have low precision.** Between 72% and 82% of flagged
claims were labelled clean by annotators, so every guard removes far more
good content than bad. Part of that is sentence-level labelling (a sentence
counts as clean unless it overlaps an annotated span), and part is that no
threshold was tuned. Even so, an operator would see most of what a guard
drops turn out to be fine.

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

**LLM verifiers.** The judge makes one call per claim with all of the
response's context chunks as numbered passages and replies in JSON; a reply
that doesn't parse counts as not supported. The self-check baseline makes
one call per response listing every sentence and asks which are
unsupported. Both use `openai/gpt-4o-mini` at temperature 0, with calls
cached by request.

**Latency and cost.** Latency is the guard's wall time per response on a
4-core CPU container with no GPU; LLM runs had eight responses in flight at
a time. NLI runs locally, so it has no API cost. LLM cost is what the
provider reported per call.

## Limitations

- **One LLM.** The judge and self-check rows use a single model
  (`gpt-4o-mini`); a stronger judge would likely change the trade-off.
- **Strict-run latency is not real.** The strict judge run reused the
  judge run's cached verdicts (the verifier calls are identical; only the
  policy differs), so its latency is cache lookups. Its cost per answer is
  that of the original calls.
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
