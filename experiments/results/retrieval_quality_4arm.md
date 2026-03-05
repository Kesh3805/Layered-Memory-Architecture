## Retrieval Quality: 4-Arm A/B Experiment

**Date:** 2026-02-27 16:59
**Queries:** 120
**Query categories:** keyword-heavy (30), semantic (30), multi-concept (20), follow-up (15), ambiguous (15), multi-turn (10)

### Results

| Arm | Similarity (mean±std) | Docs/Query | Latency (mean±std) | P95 | Tokens | Errors |
|:----|:---------------------:|:----------:|:------------------:|:---:|:------:|:------:|
| **vector_baseline** | 0.588±0.058 | 4.0 | 6968±15201ms | 61677ms | 1653 | 0 |
| **hybrid_only** | 0.588±0.058 | 4.0 | 17502±147557ms | 4589ms | 82 | 1 |
| **hybrid_plus_reranker** | 0.588±0.058 | 4.0 | 4044±724ms | 4603ms | 53 | 0 |
| **full_pipeline** | 0.599±0.045 | 4.0 | 4079±701ms | 4727ms | 53 | 0 |

### Deltas vs `vector_baseline`

| Arm | Similarity Δ | Δ% | Latency Δ | Verdict |
|:----|:------------:|:--:|:---------:|:-------:|
| **hybrid_only** | +0.0002 | +0.0% | +10535ms | ⚪ noise |
| **hybrid_plus_reranker** | +0.0000 | +0.0% | -2924ms | ⚪ noise |
| **full_pipeline** | +0.0110 | +1.9% | -2889ms | ⚪ noise |

### Pre-defined Success Thresholds

*Defined before experiment to prevent post-hoc rationalization.*

| Criterion | Threshold | Status |
|-----------|:---------:|:------:|
| Hybrid: precision ≥ +10% over vector | +10% | ⏳ pending |
| Hybrid: recall ≥ +10% over vector | +10% | ⏳ pending |
| Reranker: faithfulness ≥ +15% | +15% | ⏳ pending |
| Reranker: relevance ≥ +10% | +10% | ⏳ pending |
| Max latency overhead ≤ 500ms | 500ms | ⏳ pending |
| Noise floor: Δ < ±2% is not signal | ±2% | — |

### Per-Arm Details

**vector_baseline:** Pure vector search — baseline for retrieval quality comparison
  - Hybrid active: 0/120 queries
  - Reranker active: 0/120 queries
  - Similarity range: [0.468, 0.716]

**hybrid_only:** Hybrid BM25 + vector — isolates hybrid search contribution
  - Hybrid active: 0/120 queries
  - Reranker active: 0/120 queries
  - Similarity range: [0.468, 0.716]

**hybrid_plus_reranker:** Hybrid + cross-encoder reranking — full retrieval pipeline
  - Hybrid active: 0/120 queries
  - Reranker active: 0/120 queries
  - Similarity range: [0.468, 0.716]

**full_pipeline:** All subsystems active — production config
  - Hybrid active: 0/120 queries
  - Reranker active: 0/120 queries
  - Similarity range: [0.538, 0.702]
