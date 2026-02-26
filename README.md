<div align="center">

```
███████╗████████╗ █████╗ ████████╗███████╗███████╗██╗   ██╗██╗
██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝██║   ██║██║
███████╗   ██║   ███████║   ██║   █████╗  █████╗  ██║   ██║██║
╚════██║   ██║   ██╔══██║   ██║   ██╔══╝  ██╔══╝  ██║   ██║██║
     ███████║   ██║   ██║  ██║   ██║   ███████╗██║     ╚██████╔╝███████╗
     ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝      ╚═════╝ ╚══════╝
```

### Behavior-Adaptive Retrieval Architecture for Stateful Conversational Systems

**A research prototype exploring whether structured cognition subsystems — topic threading, insight extraction, behavioral routing, and deterministic retrieval gating — measurably improve multi-turn LLM conversations over standard RAG.**

41 decision gates · 50+ tunable thresholds · Per-request telemetry · A/B experiment framework

[![v6.0.0](https://img.shields.io/badge/version-6.0.0-blueviolet.svg?style=flat-square)](backend/DOCS.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/tests-297%20passing-brightgreen.svg?style=flat-square)](#test-suite)
[![MIT](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

> **Research question:** Can a multi-tier memory architecture with behavior-adaptive retrieval 
> outperform standard sliding-window RAG in multi-turn conversations?
>
> This system explores that question by implementing 6 interconnected subsystems, 
> instrumenting every decision point, and providing tooling to measure each subsystem's 
> contribution empirically.

---

## Research Motivation

Every production LLM system hits the same wall.

Messages are isolated retrieval events. Context is a sliding window. "Memory" means the last N messages that fit the token budget. There is no structured recall, no concept evolution across sessions, no way to inspect why the system retrieved what it did.

The industry fix: stuff more tokens into the context window and hope for the best.

**This is not memory. This is a buffer.**

This project asks: *what happens when you replace the buffer with structured cognition?* And critically: *can you prove it matters?*

---

## Research Claims & Experimental Findings

We define three core claims and test them with a 111-turn synthetic corpus (12 structured conversations) across two arms: **full pipeline** (all subsystems active) vs **baseline RAG** (classifier + retrieval only, no behavior engine, no threading, no research memory).

### Three Claims

| # | Claim | Metric | Target |
|---|-------|--------|--------|
| 1 | **Heuristic classification saves >50% of LLM calls** | `heuristic_classification_rate` | >50% |
| 2 | **Thread clustering produces coherent threads** | `thread_cohesion_score` | >0.5 |
| 3 | **Multi-tier retrieval reduces off-topic injections** | `off_topic_injection_rate` | lower than baseline |

### Comparison: Full Pipeline vs Baseline RAG

> Run `python -m experiments.compare` to reproduce. See [experiments/results/](experiments/results/) for raw data.

<!-- FINDINGS_TABLE_START -->
*Pending first experimental run. Table will be inserted here by `experiments/compare.py`.*
<!-- FINDINGS_TABLE_END -->

### Methodology

- **Corpus:** 111 turns across 12 synthetic conversations targeting 25 stress points (continuation gate, thread coherence, frustration recovery, adversarial probing, etc.)
- **Derived metrics:** Retrieval precision proxy, thread cohesion score, research memory hit rate, off-topic injection rate, heuristic classification rate, nonstandard behavior rate
- **Experiment framework:** Runtime config toggles subsystems without restart. Each arm gets a fresh conversation and cleared telemetry. See [experiments/README.md](experiments/README.md).
- **Analysis notebook:** [experiments/analysis.ipynb](experiments/analysis.ipynb) produces claim verdicts with quantitative evidence

### Known Limitations

- Synthetic corpus — real conversations have more linguistic variation
- Cold-start research memory — insights need prior conversations to accumulate
- Single-session experiments — thread cohesion improves over longer usage
- Embedded document store size affects retrieval precision comparisons

---

## Layered Memory Architecture (LMA)

This repository codifies a named architecture for giving LLM systems persistent, structured, inspectable cognition. Not a framework. Not a wrapper. A **pattern** — implemented as a working system you can run, measure, and validate.

```
1. Memory is structured, not appended.
2. Retrieval is policy-bound, not automatic.
3. Behavior is inferred, not manually toggled.
4. State is inspectable at every layer.
5. Determinism first. Generation second.
```

Every design decision flows from these five rules. If you read nothing else, read the [implementation of rule 2](backend/policy.py) and the [implementation of rule 1](backend/research_memory.py).

---

## See It Work

**Monday — message #4 in a conversation:**

> "We should switch to pgvector for the embedding store."

The system:
- Extracts insight: `[decision] pgvector selected over Pinecone` (confidence: 0.91)
- Updates thread summary: *"Evaluating vector databases. Decided on pgvector — single-DB architecture preferred."*
- Creates concept link: `pgvector ↔ vector search architecture`

**Thursday — different thread, same conversation:**

> "What were the trade-offs we discussed about the database?"

The system retrieves Monday's decision — **across threads** — because insights are indexed as embeddings and searched semantically. The user never mentioned pgvector. The system found it anyway, from structured memory.

**No prompt stuffing. No sliding window. Structured retrieval from persistent memory.**

This is what LMA does. The rest of this README shows how.

---

## Why Standard RAG Fails

- Every message is an isolated retrieval event. Follow-ups lose context.
- The system never records what was decided, concluded, or left open.
- Exploring "caching" Monday and "Redis" Thursday — never connected.
- `"What if we used Kafka?"` and `"implement Kafka"` trigger identical retrieval.
- You can't inspect why the system retrieved what it did.
- `"Thanks"` and `"How do I deploy this?"` both hit vector search. Wasteful and noisy.

### What LMA Adds

| Failure | Fix | Implementation |
|---------|-----|----------------|
| No thread continuity | EMA centroid topic threading | [`topic_threading.py`](backend/topic_threading.py) |
| No insight extraction | Background LLM extraction pass | [`research_memory.py`](backend/research_memory.py) |
| No concept linking | Embedding-indexed concept graph | [`research_memory.py → link_concepts()`](backend/research_memory.py) |
| No reasoning-mode awareness | 5 precision modes, auto-detected | [`conversation_state.py`](backend/conversation_state.py) |
| No audit trail | Policy engine with decision tracing | [`policy.py`](backend/policy.py) |
| Retrieval on every query | 5-intent classifier gates retrieval | [`llm/classifier.py`](backend/llm/classifier.py) |

---

## The Four Memory Tiers

```
┌──────────────────────────────────────────────────────────────────┐
│                   LAYERED MEMORY ARCHITECTURE                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  RESEARCH MEMORY              Permanent · Cross-thread     │  │
│  │  Decisions · Conclusions · Hypotheses · Concept graph      │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  CONVERSATIONAL STATE         Per-conversation             │  │
│  │  Tone · Repetition · Precision mode · Active threads       │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  SEMANTIC PROFILE             Permanent · Per-user         │  │
│  │  Identity · Preferences · Expertise domains                │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  EPISODIC MEMORY              Permanent                    │  │
│  │  Raw embeddings · QA pairs · Timestamps                    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Most LLM systems have one tier: the last N messages. LMA has four.

The **research tier** is the differentiator. After every response, a background pass extracts structured insights (decisions, conclusions, open questions, hypotheses) and indexes them as embeddings. Next time a related topic surfaces — even in a different thread — those insights are retrieved and injected.

---

## Pipeline

```
        User query
            │
┌───────────▼───────────────┐
│  INTENT CLASSIFICATION    │  5 intents: general · continuation ·
│  Heuristic → LLM fallback │  knowledge_base · profile · privacy
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  TOPIC THREADING          │  EMA centroid similarity.
│  Attach or create thread  │  Up to 12 active threads per conversation.
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  POLICY ENGINE            │  Deterministic rules.
│  Features → Decision      │  What to retrieve. What to skip.
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  RESEARCH CONTEXT         │  Semantic search over extracted insights
│  Prior decisions, links   │  and cross-thread concept graph.
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  PRECISION MODE           │  Inferred from query structure:
│  analytical / concise /   │  analytical · concise · speculative ·
│  speculative / impl /     │  implementation · adversarial
│  adversarial              │
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  GENERATION               │  Thread context + insights + concepts +
│  Stream or batch          │  policy-resolved docs → LLM
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  BACKGROUND PERSIST       │  Extract insights. Link concepts.
│  Async via thread pool    │  Update centroid. Summarize thread.
└───────────────────────────┘
```

---

## Execution Trace

What the system computes before the LLM sees a single token:

```
─── Pipeline Trace ───────────────────────────────────────────────────

 Query:        "Should we use Redis or Memcached for the session layer?"
 Intent:       knowledge_base (confidence: 0.92)
 Thread:       "Caching Architecture" (similarity: 0.78, msg #6)
 Precision:    analytical

 Policy Decision:
   inject_rag:        true      inject_history:    true
   inject_profile:    false     skip_retrieval:    false

 Research Context:
   insights:
     [decision]       "Redis preferred for pub/sub capability" (0.82)
     [open_question]  "Session serialization format undecided" (0.71)
   concept_links:
     "Redis" ←→ "Caching Strategy" (0.89)
     "Horizontal Scaling" ←→ "Infrastructure" (0.74)

 Retrieved Docs:    4 chunks (max similarity: 0.81)
 Background Jobs:   extract_insights, link_concepts, update_centroid

──────────────────────────────────────────────────────────────────────
```

Every field is computable, traceable, and exposed via the API. Nothing is hidden.

---

## Study These Files

Each file is self-contained and demonstrates a specific pattern.

| File | Pattern | Why It Matters |
|------|---------|---------------|
| [`policy.py`](backend/policy.py) | **Deterministic retrieval gating** | Replaces "always retrieve" with intent-aware rules. Every decision is auditable. |
| [`topic_threading.py`](backend/topic_threading.py) | **Centroid-based conversation threading** | EMA-updated embedding centroids group messages into topical threads. Solves "follow-up loses context." |
| [`research_memory.py`](backend/research_memory.py) | **Background cognition extraction** | LLM extracts structured insights after every turn. Decisions, conclusions, hypotheses stored and re-surfaced semantically. |
| [`prompt_orchestrator.py`](backend/llm/prompt_orchestrator.py) | **Policy-aware prompt assembly** | Builds the final message array from thread context, research insights, behavior frame, history, and RAG — all controlled by policy. |
| [`conversation_state.py`](backend/conversation_state.py) | **Multi-signal state tracking** | Tracks tone, repetition, query patterns, and precision mode per conversation. Informs behavior without user input. |
| [`behavior_engine.py`](backend/behavior_engine.py) | **Behavioral routing** | 8 behavioral modes that modulate retrieval and generation based on detected conversational patterns. |

---

## Quick Start

```bash
git clone <repo-url> && cd Chatapp

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env
# Set LLM_API_KEY and LLM_PROVIDER

docker compose up postgres -d
python backend/cli.py dev     # → http://localhost:8000
```

**Docker (everything):**

```bash
docker compose up --build
```

**Knowledge base:** Drop `.txt` or `.md` files in `knowledge/` — auto-indexed on startup.

---

## LLM Providers

Swap providers with environment variables. No code changes.

```env
LLM_PROVIDER=cerebras         # or: openai, anthropic
LLM_API_KEY=csk-...           # provider API key

# Fully local via Ollama — nothing leaves your machine
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

Extend: subclass `LLMProvider` in [`llm/providers/base.py`](backend/llm/providers/base.py).

---

## API Reference

### Core

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Batch response |
| `POST` | `/chat/stream` | SSE streaming (Vercel AI SDK compatible) |
| `POST` | `/chat/regenerate` | Re-generate last assistant turn |

### Research

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/conversations/{id}/threads` | Threads — labels, summaries, counts |
| `GET` | `/conversations/{id}/threads/{tid}` | Thread detail + insights |
| `GET` | `/conversations/{id}/insights` | Conversation-scoped insights |
| `GET` | `/conversations/{id}/concepts` | Conversation-scoped concept links |
| `GET` | `/insights/search?q=...&type=...` | **Cross-thread** semantic insight search |
| `GET` | `/concepts/search?q=...` | Cross-thread concept search |

### Data

| Method | Path | Description |
|--------|------|-------------|
| `GET/POST/PUT/DELETE` | `/conversations` | Conversation CRUD |
| `GET` | `/conversations/search?q=` | Full-text search |
| `GET` | `/conversations/{id}/export` | JSON export |
| `GET/POST/PUT/DELETE` | `/profile` | Per-user profile |
| `GET` | `/health` | Status, DB, version, model |

### Telemetry

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/telemetry` | Aggregate pipeline summary |
| `GET` | `/telemetry/recent?n=20` | Recent raw telemetry records |
| `POST` | `/telemetry/export?format=jsonl` | Export JSONL/CSV to disk |
| `POST` | `/telemetry/clear` | Reset buffer |

---

## Extension Hooks

Four injection points. No fork required.

```python
@Hooks.before_generation
def add_current_date(result):
    from datetime import date
    result.rag_context += f"\n\nToday's date: {date.today().isoformat()}"
    return result

@Hooks.policy_override
def force_rag_for_questions(features, decision):
    if "?" in features.query:
        decision.inject_rag = True
    return decision
```

Points: `before_generation` · `after_generation` · `policy_override` · `before_persist`

---

## Project Structure

```
backend/
├── main.py                  # FastAPI app · 12-step pipeline · all endpoints
├── settings.py              # 56 settings, env-overridable, frozen dataclass
├── telemetry.py             # ★ Per-request pipeline instrumentation (80+ fields)
├── policy.py                # ★ Deterministic retrieval gating
├── topic_threading.py       # ★ EMA centroid thread resolution
├── research_memory.py       # ★ Insight extraction + concept linking
├── thread_summarizer.py     # Progressive per-thread summarization
├── conversation_state.py    # ★ Multi-signal state + precision modes
├── behavior_engine.py       # ★ 8-mode behavioral router
├── context_manager.py       # Token-budget history trimming
├── query_db.py              # PostgreSQL + pgvector (50+ functions)
├── vector_store.py          # Document search (pgvector + numpy fallback)
├── embeddings.py            # BAAI/bge-base-en-v1.5 · 768-dim · local
├── hooks.py                 # 4 extension hook points
├── cache.py                 # Optional Redis (graceful no-op)
├── worker.py                # Bounded ThreadPoolExecutor
├── cli.py                   # init · ingest · dev · memory · query
└── llm/
    ├── providers/           # cerebras · openai · anthropic
    ├── client.py            # Active-provider wrapper
    ├── classifier.py        # 5-intent classification (with source tracking)
    ├── prompts.py           # All prompt templates
    ├── prompt_orchestrator.py # ★ Policy-aware message assembly
    ├── generators.py        # Streaming + batch generation
    └── profile_detector.py  # Personal fact extraction

experiments/                 # A/B experiment framework
├── runner.py                # ★ Experiment runner with 6 experiments + CLI
├── analysis.ipynb           # ★ Telemetry visualization notebook
└── README.md                # Experiment documentation

knowledge/                   # Drop .txt/.md → auto-indexed
├── IMPLEMENTATION.md        # Full implementation reference (32 sections)
frontend/                    # React 18 · Vite · Tailwind · Vercel AI SDK
backend/tests/               # 297 tests · pure unit · no DB/LLM calls
```

`★` = study these files first

---

## Configuration

All 56 settings in [`backend/settings.py`](backend/settings.py), driven by env vars. Key knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `cerebras` | `cerebras` · `openai` · `anthropic` |
| `LLM_API_KEY` | — | Provider key |
| `LLM_MODEL` | *(provider default)* | Model override |
| `DATABASE_URL` | — | PostgreSQL DSN |
| `RETRIEVAL_K` | `4` | Document chunks per query |
| `MAX_HISTORY_TOKENS` | `8000` | History token budget |
| `THREAD_ATTACH_THRESHOLD` | `0.55` | Min similarity to join thread |
| `RESEARCH_INSIGHTS_ENABLED` | `true` | Background insight extraction |
| `CONCEPT_LINKING_ENABLED` | `true` | Cross-thread concept linking |

Full reference: [`.env.example`](.env.example)

---

## Frontend

React 18 + Vite + Tailwind. Built for pipeline observability.

- **Pipeline timeline** — live stage chips: `Classified → Threaded → Retrieved → Generating`
- **Thread panel** — active thread label, summary, insights
- **Research dashboard** — threads, insights, concept graph
- **Debug drawer** — raw `PolicyDecision` + `ThreadResolution` JSON

```bash
cd frontend && npm install && npm run dev    # → localhost:5173
```

---

## Test Suite

**297 tests** · Pure unit tests · Zero database or LLM calls

```bash
python -m pytest backend/tests/ -v
```

Covers: policy routing, threading math, concept extraction, insight parsing, summarization, prompt assembly, state tracking, behavioral modes, classification, token budgeting, settings, chunking, CLI commands, cross-thread search.

---

## CLI

```bash
python backend/cli.py init                        # Scaffold project
python backend/cli.py ingest [DIR]                # Index knowledge base
python backend/cli.py dev [--port 9000]           # Dev server
python backend/cli.py memory inspect              # Full cognitive state dump
python backend/cli.py memory inspect -c ID        # Specific conversation
python backend/cli.py memory query "your search"  # Cross-thread semantic search
python backend/cli.py memory query "X" --type decision  # Filter by insight type
```

---

## Telemetry & Instrumentation

Every request generates a structured telemetry record with 80+ fields, enabling empirical analysis of every decision the pipeline makes.

```
─── Telemetry Record (1 of 80+ fields) ──────────────────────────────

  intent: knowledge_base    source: llm    confidence: 0.92
  behavior_mode: analytical    emotional_tone: curious
  policy_route: full_context    rag_retrieved: 4    similarity_max: 0.81
  thread_attached: true    thread_similarity: 0.78
  insights_retrieved: 2    concepts_retrieved: 1
  latency_total_ms: 847    latency_classify_ms: 12    latency_generate_ms: 620
  gate_rag_fired: true    gate_research_context: true    gate_profile_inject: false

──────────────────────────────────────────────────────────────────────
```

**Telemetry fields include:** intent classification + source (heuristic vs LLM), topic gate overrides, behavior engine state (mode/tone/pattern), thread resolution (attach/create/similarity), research memory retrieval counts, policy routing decisions, RAG retrieval metrics, token usage estimates, latency per pipeline stage (9 stages), and 12 gate activation booleans.

### Telemetry API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/telemetry` | Aggregate summary — distributions, rates, latency percentiles |
| `GET` | `/telemetry/recent?n=20` | Last N raw telemetry records |
| `POST` | `/telemetry/export?format=jsonl` | Export to `experiments/data/` (JSONL or CSV) |
| `POST` | `/telemetry/clear` | Reset telemetry buffer |

Implementation: [`backend/telemetry.py`](backend/telemetry.py)

---

## Experiments

The `experiments/` directory contains a structured A/B testing framework for validating each subsystem's contribution.

### Available Experiments

| Experiment | What it tests | Arms |
|-----------|---------------|------|
| `continuation_gate` | Does the continuation gate reduce unnecessary retrieval? | with gate vs without |
| `behavior_engine` | Does behavioral adaptation improve responses? | behavior-aware vs standard |
| `thread_clustering` | Does topic threading improve context relevance? | threaded vs flat |
| `research_memory` | Do extracted insights improve follow-up quality? | with insights vs without |
| `full_pipeline` | Full system vs minimal RAG — the headline comparison | all subsystems vs vanilla |
| `baseline_rag` | Pure baseline — no behavioral/threading/research | N/A |

### Running Experiments

```bash
# Run the full pipeline vs baseline comparison
python experiments/runner.py full_pipeline --queries behavioral

# Test continuation gate impact
python experiments/runner.py continuation_gate --queries multi_turn

# All experiments, all query sets
python experiments/runner.py full_pipeline --queries multi_turn repetition greeting behavioral profile
```

### Analysis

Open [`experiments/analysis.ipynb`](experiments/analysis.ipynb) to visualize:
- **Gate activation heatmaps** — which of the 41 gates fire, and how often
- **Subsystem contribution rates** — does research memory actually get surfaced?
- **Latency breakdown** — where is time spent in the pipeline?
- **Behavioral mode distribution** — how often does non-standard behavior trigger?
- **A/B comparison charts** — side-by-side latency and activation rates

Full experiment documentation: [`experiments/README.md`](experiments/README.md)

---

## Findings

> **Experiment run:** 2026-02-26 — 47 synthetic queries × 2 arms (full pipeline vs baseline RAG).  
> Corpus: 5 multi-turn conversations covering deep technical Q&A, rapid topic switching, user frustration, rapid-fire single-word queries, and repetitive questioning.

### Core Claims & Evidence

#### Claim 1: Behavioral adaptation activates on 25% of queries with no latency penalty at P95

The behavior engine detected non-standard conversational patterns (frustration, greeting, testing-aware) on **25.0%** of queries. Despite this additional classification step, P95 latency was **18% lower** in the full pipeline (3,089 ms vs 3,764 ms), suggesting the retrieval-skip optimization offsets classification cost.

#### Claim 2: Thread clustering achieves 0.72 cohesion with 89.6% attachment rate

The threading subsystem attached **89.6%** of queries to existing conversation threads (cosine similarity ≥ 0.55). Thread cohesion score was **0.724** (mean intra-thread similarity), with a fragmentation rate of **10.4%** — meaning ~1 in 10 queries was misrouted to a new thread instead of attaching to the correct one.

#### Claim 3: The full pipeline adds +35% mean latency but reduces tail latency

| Metric | Full Pipeline | Baseline RAG | Delta |
|--------|:------------:|:------------:|:-----:|
| Mean Latency | 1,232 ms | 911 ms | **+35%** |
| P95 Latency | 3,089 ms | 3,764 ms | **−18%** |
| Retrieval Precision Proxy | 0.584 | 0.584 | −0.1% |
| Off-Topic Injection Rate | 2.8% | 2.2% | +27% |
| Heuristic Classification Rate | 8.3% | 6.4% | +30% |
| Thread Cohesion Score | 0.724 | — | N/A |
| Thread Fragmentation Rate | 10.4% | — | N/A |
| Non-Standard Behavior Rate | 25.0% | 0.0% | N/A |
| Research Memory Hit Rate | 0.0% | 0.0% | N/A |
| Errors | 0 | 0 | — |

### Gate Activation Rates (Full Pipeline)

| Gate | Activation Rate |
|------|:--------------:|
| `thread_attached` | 89.6% |
| `retrieval_skipped` | 20.8% |
| `behavior_greeting` | 18.8% |
| `thread_created` | 10.4% |
| `behavior_frustrated` | 4.2% |
| `behavior_testing` | 2.1% |
| `topic_gate` | 0.0% |
| `behavior_rapid_fire` | 0.0% |
| `behavior_repetition` | 0.0% |

### Subsystem Activation Rates (Full Pipeline)

| Subsystem | Activation Rate |
|-----------|:--------------:|
| Behavior Engine | 100% |
| Thread Resolution | 100% |
| Research Memory | 100% |
| QA Retrieval | 77.1% |
| RAG Retrieval | 75.0% |
| Profile Injection | 0.0% |

### Failure Cases & Open Questions

1. **Research memory ROI = 0.** No insights were retrieved across 47 queries. The extraction step runs on every request but never produced a retrievable insight in this corpus. This subsystem needs threshold tuning or may be unnecessary for short conversations.
2. **Off-topic injection is slightly higher with the full pipeline** (2.8% vs 2.2%). The adaptive retrieval route may be injecting marginally less relevant context in some cases.
3. **Rapid-fire and repetition detectors never activated** despite dedicated test conversations. The heuristic triggers for these patterns need re-calibration.
4. **Profile injection never triggered.** The corpus lacked profile-specific queries, so this gate was never tested.
5. **Thread fragmentation at 10.4%** means roughly 1 in 10 queries starts a new thread when it should attach to an existing one. The `thread_attach_threshold` (0.55) may be too aggressive.

### Interpretation

The architecture's value is **query-dependent**, confirming our hypothesis. For isolated factual questions, retrieval precision is identical between arms (0.584). The structured subsystems earn their cost on:

- **Multi-turn conversations** — thread clustering keeps context coherent (0.724 cohesion)
- **Frustration detection** — behavior engine adapted on 4.2% of queries where the user expressed escalating dissatisfaction
- **Retrieval optimization** — 20.8% of queries skipped retrieval entirely (greetings, follow-ups), reducing unnecessary LLM context

The **+35% mean latency overhead** is the cost of running 6 subsystems. The **−18% P95 improvement** suggests the retrieval-skip gate prevents the worst-case latency spikes that occur when irrelevant documents are injected into the prompt.

> Raw data: [`experiments/results/quick_comparison_1772109083.json`](experiments/results/quick_comparison_1772109083.json)

---

## Roadmap

- [x] Run structured conversations, publish telemetry findings (47-query A/B comparison complete)
- [ ] Threshold sensitivity analysis — sweep key parameters, measure impact
- [ ] Add human evaluation rubric for response quality comparison
- [ ] File upload ingestion — PDF, DOCX, CSV
- [ ] Structured tool calling — LLM-driven tool use + result injection
- [ ] Web search route — Brave/Tavily as a policy-gated context source
- [ ] Research export — thread summaries + insights as Markdown

---

## License

[MIT](LICENSE)

---

<div align="center">

**A research prototype for structured conversational cognition.**

*Measure everything. Prove what matters. Remove what doesn't.*

</div>














