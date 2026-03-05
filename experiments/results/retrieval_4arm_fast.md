## Retrieval Quality: 4-Arm A/B Experiment

**Date:** 2026-03-05 15:48
**Queries per arm:** 80
**Total queries:** 320
**Method:** Pure retrieval (no LLM generation) — `/retrieval/test` endpoint

### Pre-Defined Success Thresholds

*Set before running to prevent post-hoc rationalisation.*

| Criterion | Threshold |
|-----------|:---------:|
| Hybrid Δsimilarity ≥ noise floor | +0.020 |
| Hybrid doc diversity ≥ | 10% |
| Reranker Δsimilarity ≥ noise floor | +0.020 |
| Max added latency per query | 500 ms |
| Noise floor (similarity) | ±0.005 |
| Noise floor (latency) | ±50 ms |

### Results

| Arm | Similarity (mean±std) | Latency (mean±std) | P95 | Docs/q | Errors |
|:----|:---------------------:|:------------------:|:---:|:------:|:------:|
### Results

| Arm | Similarity (mean±std) | Latency ms (mean±std) | P95 ms | Docs/q | Errors |
|:----|:---------------------:|:---------------------:|:------:|:------:|:------:|
| **vector_baseline** | 0.6097±0.0566 | 2236±127 | 2412 | 4.0 | 0 |
| **hybrid_only** | 0.6026±0.0705 | 2191±32 | 2244 | 4.0 | 0 |
| **hybrid_plus_reranker** | 0.5900±0.0667 | 2787±224 | 3171 | 4.0 | 0 |
| **full_pipeline** | 0.5900±0.0667 | 2988±1223 | 3603 | 4.0 | 0 |

### Deltas vs `vector_baseline`

| Arm | Δ Similarity | Δ Latency ms | Doc diversity | Verdict |
|:----|:------------:|:------------:|:-------------:|:-------:|
| **hybrid_only** | -0.0071 | -44 | 8.7% | ❌ regression |
| **hybrid_plus_reranker** | -0.0197 | +551 | 95.0% | ❌ regression |
| **full_pipeline** | -0.0197 | +753 | 95.0% | ❌ regression |

### Threshold Checklist

*Each criterion was defined before running.*

| Criterion | Threshold | Measured | Status |
|-----------|:---------:|:--------:|:------:|
| Hybrid Δsimilarity | ≥0.020 | -0.0071 | ❌ FAIL |
| Hybrid doc diversity | ≥10% | 8.7% | ❌ FAIL |
| Hybrid latency delta | ≤500 ms | -44 ms | ✅ PASS |
| Reranker Δsimilarity | ≥0.020 | -0.0126 | ❌ FAIL |
| Reranker latency delta | ≤500 ms | +595 ms | ❌ FAIL |

### Engineering Interpretation

> **Hybrid search:** Δsimilarity = -0.0071 — a regression. Hybrid search is performing WORSE than pure vector on this corpus.
> **Reranker:** Δsimilarity = -0.0126 — regression vs hybrid-only.

> **Caveat:** Cosine similarity is an imperfect proxy for retrieval quality. The gold standard metrics (faithfulness, answer relevance via LLM-as-judge) require running the full `/chat` pipeline. These results establish the retrieval-only baseline; production recommendation should be verified with end-to-end quality metrics.