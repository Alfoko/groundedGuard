# judge-strict

RAGTruth `test` split, n = 300 responses, seed 13. Intervals are 95% bootstrap CIs over responses.

| Metric | Value |
|---|---|
| hallucinated claim rate | 1.4% [0.7, 2.2] |
| answer hallucination rate | 8.2% [4.1, 12.9] |
| over refusal rate | 51.3% [45.7, 57.0] |
| verifier precision | 28.0% [19.7, 37.2] |
| verifier recall | 74.6% [64.7, 82.2] |
| latency p50 / p95 | 0.014 s / 0.036 s |
| cost per answer | $0.00110 |
