# RAGTruth comparison

n = 300 test responses shared by every run. Brackets are 95% bootstrap CIs over responses; p-values are two-sided paired permutation tests against `baseline` on the same responses.

| System | Hallucinated claim rate | Answer hallucination rate | Over-refusal rate | Latency p50 | Cost / answer |
|---|---|---|---|---|---|
| baseline | 8.3% [6.0, 11.0] | 32.0% [26.7, 37.3] | 0.0% [0.0, 0.0] | — | — |
| judge | 2.6% [1.7, 3.4] (p < 0.001) | 13.3% [9.4, 17.4] (p < 0.001) | 9.7% [6.3, 13.0] (p < 0.001) | 8.35 s | $0.0011 |
| judge-strict | 1.4% [0.7, 2.2] (p < 0.001) | 8.2% [4.1, 12.9] (p < 0.001) | 51.3% [45.7, 57.0] (p < 0.001) | 0.01 s | $0.0011 |
| nli | 2.3% [1.5, 3.2] (p < 0.001) | 11.1% [7.3, 15.1] (p < 0.001) | 15.7% [11.7, 19.7] (p < 0.001) | 4.46 s | $0.0000 |
| self_check | 5.0% [2.3, 8.3] (p < 0.001) | 10.5% [7.1, 14.2] (p < 0.001) | 5.0% [2.7, 7.7] (p < 0.001) | 0.72 s | $0.0001 |
