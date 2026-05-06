## Retrieval Quality: 4-Arm A/B Experiment

**Date:** 2026-05-06 11:19
**Queries per arm:** 10
**Total queries:** 40
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

| Arm | Similarity (mean±std) | Latency ms (mean±std) | P95 ms | Docs/q | Errors |
|:----|:---------------------:|:---------------------:|:------:|:------:|:------:|
| **vector_baseline** | 0.6426±0.0402 | 2254±23 | 2286 | 4.0 | 0 |
| **hybrid_only** | 0.6426±0.0402 | 2312±45 | 2385 | 4.0 | 0 |
| **hybrid_plus_reranker** | 0.6295±0.0429 | 3933±3752 | 9314 | 4.0 | 0 |
| **full_pipeline** | 0.6295±0.0429 | 2740±64 | 2846 | 4.0 | 0 |

### Deltas vs `vector_baseline`

| Arm | Δ Similarity | Δ Latency ms | Doc diversity | Verdict |
|:----|:------------:|:------------:|:-------------:|:-------:|
| **hybrid_only** | +0.0000 | +57 | 0.0% | ⚪ noise |
| **hybrid_plus_reranker** | -0.0131 | +1678 | 90.0% | ❌ regression |
| **full_pipeline** | -0.0131 | +486 | 90.0% | ❌ regression |

### Threshold Checklist

*Each criterion was defined before running.*

| Criterion | Threshold | Measured | Status |
|-----------|:---------:|:--------:|:------:|
| Hybrid Δsimilarity | ≥0.020 | +0.0000 | ❌ FAIL |
| Hybrid doc diversity | ≥10% | 0.0% | ❌ FAIL |
| Hybrid latency delta | ≤500 ms | +57 ms | ✅ PASS |
| Reranker Δsimilarity | ≥0.020 | -0.0131 | ❌ FAIL |
| Reranker latency delta | ≤500 ms | +1621 ms | ❌ FAIL |

### Engineering Interpretation

> **Hybrid search:** Δsimilarity = +0.0000 (noise floor ±0.005). On this 10-query corpus, hybrid and pure vector retrieve identical documents 100% of the time. BM25 + vector fusion adds no measurable quality improvement over pure vector on a 80-document knowledge base.
> **Reranker:** Δsimilarity = -0.0131 — regression vs hybrid-only.

> **Caveat:** Cosine similarity is an imperfect proxy for retrieval quality. The gold standard metrics (faithfulness, answer relevance via LLM-as-judge) require running the full `/chat` pipeline. These results establish the retrieval-only baseline; production recommendation should be verified with end-to-end quality metrics.