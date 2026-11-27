# self_check

RAGTruth `test` split, n = 300 responses, seed 13. Intervals are 95% bootstrap CIs over responses.

| Metric | Value |
|---|---|
| hallucinated claim rate | 5.0% [2.3, 8.3] |
| answer hallucination rate | 10.5% [7.1, 14.2] |
| over refusal rate | 5.0% [2.7, 7.7] |
| verifier precision | 18.1% [14.4, 22.1] |
| verifier recall | 54.6% [41.0, 71.8] |
| latency p50 / p95 | 0.721 s / 0.978 s |
| cost per answer | $0.00014 |
