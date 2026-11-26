# RAGTruth comparison

n = 300 test responses shared by every run. Brackets are 95% bootstrap CIs over responses; p-values are two-sided paired permutation tests against `baseline` on the same responses.

| System | Hallucinated claim rate | Answer hallucination rate | Over-refusal rate | Latency p50 | Cost / answer |
|---|---|---|---|---|---|
| baseline | 8.3% [6.0, 11.0] | 32.0% [26.7, 37.3] | 0.0% [0.0, 0.0] | — | — |
| nli | 2.3% [1.5, 3.2] (p < 0.001) | 11.1% [7.3, 15.1] (p < 0.001) | 15.7% [11.7, 19.7] (p < 0.001) | 4.46 s | $0.0000 |
