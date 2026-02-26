# Layered Memory Architecture (LMA) — System Design

> **Version 6.0.0** — This is how serious LLM systems should be built.

A reference implementation of the Layered Memory Architecture for LLM systems. Multi-tier persistent memory, deterministic retrieval routing, background cognition extraction, and full pipeline observability.

---

## Design Principles

```
1. Memory is structured, not appended.
2. Retrieval is policy-bound, not automatic.
3. Behavior is inferred, not manually toggled.
4. State is inspectable at every layer.
5. Determinism first. Generation second.
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   React Frontend (frontend/)                     │
│  Pipeline timeline · Thread panel · Research dashboard · Debug   │
│  Vite + Tailwind + Vercel AI SDK (useChat) + Zustand             │
└──────────────────────┬───────────────────────────────────────────┘
                       │  HTTP / SSE (Vercel AI SDK data stream)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI  (main.py)  v6.0.0                    │
│                                                                  │
│  POST /chat/stream   ──► run_pipeline() ──► hooks ──► generate   │
│  POST /chat          │   24 routes total                         │
│  GET /health         │   Conversations + Research + Profile CRUD │
│  GET /               │   Serve React or fallback HTML            │
└─────────┬───────────┴───┬────────┬───────────┬───────────────────┘
          │               │        │           │
  ┌───────▼───────┐ ┌────▼─────┐ ┌▼─────────┐ ┌▼──────────────────┐
  │  llm/ package │ │ query_db │ │vector_   │ │ LMA Subsystems    │
  │  ├ providers/ │ │PostgreSQL│ │store     │ │ ├ topic_threading  │
  │  │ ├ cerebras│ │ pgvector │ │ pgvector │ │ ├ research_memory  │
  │  │ ├ openai  │ │ 9 tables │ │ + numpy  │ │ ├ behavior_engine  │
  │  │ └ anthrop.│ └──────────┘ │ fallback │ │ ├ conversation_    │
  │  ├ client    │              └──────────┘ │ │   state           │
  │  ├ classifier│                           │ ├ thread_summarizer│
  │  ├ orchestr. │  ┌──────────┐ ┌────────┐ │ └ policy           │
  │  ├ generators│  │ hooks.py │ │settings│ └──────────────────────┘
  │  └ profiler  │  │ 4 hooks  │ │56 vars │
  └──────────────┘  └──────────┘ └────────┘
```

---

## The Four Memory Tiers

The core pattern. Most LLM systems have one tier (last N messages). LMA has four.

```
┌──────────────────────────────────────────────────────────────────┐
│                   LAYERED MEMORY ARCHITECTURE                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  RESEARCH MEMORY              Permanent · Cross-thread     │  │
│  │  Decisions · Conclusions · Hypotheses · Concept graph      │  │
│  │  Tables: research_insights, concept_links                  │  │
│  │  Code: research_memory.py                                  │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  CONVERSATIONAL STATE         Per-conversation             │  │
│  │  Tone · Repetition · Precision mode · Active threads       │  │
│  │  Tables: conversation_state, conversation_threads          │  │
│  │  Code: conversation_state.py, topic_threading.py           │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  SEMANTIC PROFILE             Permanent · Per-user         │  │
│  │  Identity · Preferences · Expertise domains                │  │
│  │  Tables: user_profile                                      │  │
│  │  Code: llm/profile_detector.py                             │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  EPISODIC MEMORY              Permanent                    │  │
│  │  Raw embeddings · QA pairs · Timestamps                    │  │
│  │  Tables: user_queries, chat_messages                       │  │
│  │  Code: query_db.py                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

The **research tier** is the differentiator. After every response, a background pass extracts structured insights (decisions, conclusions, open questions, hypotheses, observations) and indexes them as embeddings. Next time a related topic surfaces — even in a different thread — those insights are retrieved and injected.

---

## Pipeline Flow (12 Steps)

```
User message arrives
       │
       ▼
  ┌─ PARALLEL (ThreadPoolExecutor) ─────────────────────────────┐
  │  1. Embed query (BAAI/bge-base-en-v1.5, 768-dim)           │
  │  2. Load history (PostgreSQL)                               │
  │  3. Load profile (PostgreSQL)                               │
  └─────────────────────────────────────────────────────────────┘
       │
       ▼
  4a. Classify intent (heuristic → LLM fallback → cache)
      5 intents: general · continuation · knowledge_base · profile · privacy
       │
       ▼
  4b. Topic similarity gate
      continuation + cosine_sim < 0.35 → downgrade to general
       │
       ▼
  4c. Behavior engine evaluation
      ConversationState → BehaviorDecision (8 modes, personality, retrieval mod)
       │
       ▼
  4d. Topic threading
      resolve_thread() → EMA centroid similarity → attach or create thread
       │
       ▼
  4e. Research context
      get_research_context() → related insights + concept links for active thread
       │
       ▼
  5a. Policy resolution
      extract_context_features() → BehaviorPolicy.resolve() → Hooks.policy_override()
       │
       ▼
  5b. Apply behavior overrides
      BehaviorDecision → adjust retrieval depth, skip/reduce/boost
       │
       ▼
  6. History pruning
      Recency window + semantic retrieval of older relevant messages
       │
       ▼
  7. Selective retrieval
      Policy-gated: RAG docs, cross-conv Q&A, same-conv Q&A, profile context
       │
       ▼
  8. Hooks.before_generation()
       │
       ▼
  9. Generate response (stream or batch via active LLM provider)
       │
       ▼
  10. Hooks.after_generation() → Hooks.before_persist()
       │
       ▼
  11-12. Background persist (worker.submit → non-blocking):
         Store messages → update topic vector (EMA, α=0.2) → auto-title
         → detect profile updates → save conversation state
         → extract research insights → link concepts → summarize thread
```

---

## Core Subsystems

### Intent Classification

Every message is classified **before** any retrieval happens.

| Intent | Retrieval | LLM Receives |
|--------|-----------|-------------|
| `general` | None | System prompt + query |
| `knowledge_base` | pgvector docs + Q&A | Docs + Q&A + history + query |
| `continuation` | Same-conv Q&A | Curated history + Q&A + query |
| `profile` (statement) | None (saved in background) | System prompt + query |
| `profile` (question) | User profile | Profile + query |
| `privacy` | Profile + rules | Profile + privacy frame + query |

Pre-heuristics bypass the LLM call for greetings, profile statements, privacy phrases, and short follow-ups.

### Policy Engine

Deterministic rules that decide what context to inject. No prompts, no LLM calls.

```python
features = extract_context_features(query, intent, profile_entries, ...)
decision = BehaviorPolicy().resolve(features, intent)
decision = Hooks.run_policy_override(features, decision)
```

The `PolicyDecision` controls: `inject_rag`, `inject_profile`, `inject_qa_history`, `privacy_mode`, `greeting_name`, `retrieval_route`, `rag_k`, `rag_min_similarity`.

### Topic Threading

EMA-updated embedding centroids group messages into topical threads.

- Up to 12 active threads per conversation
- `resolve_thread()` computes cosine similarity against all thread centroids
- Above `THREAD_ATTACH_THRESHOLD` (0.55) → attach to existing thread
- Below threshold → create new thread
- Centroids updated via exponential moving average after each message
- Threads summarized every `THREAD_SUMMARY_INTERVAL` (8) messages

### Research Memory

Background cognition extraction — the system learns from every interaction.

1. After each response, LLM extracts structured insights: `decision`, `conclusion`, `hypothesis`, `open_question`, `observation`
2. Insights stored with embeddings in `research_insights` table
3. Concept nouns extracted and linked in `concept_links` table
4. On next related query (even in a different thread), insights resurface via semantic search
5. Cross-thread concept graph connects ideas across topics

### Behavior Engine

8 modes that modulate retrieval and generation:

| Mode | Trigger | Effect |
|------|---------|--------|
| `standard` | Default | Normal RAG pipeline |
| `greeting` | Social messages | Minimal/no retrieval |
| `repetition_aware` | Repeated patterns | Vary response, acknowledge pattern |
| `testing_aware` | Probing the system | Meta-honest engagement |
| `meta_aware` | Commenting on AI | Self-aware, steer back |
| `frustration_recovery` | Frustration signals | Empathetic, thorough |
| `rapid_fire` | Short rapid messages | Concise, direct |
| `exploratory` | Open-ended exploration | Broader context, diverse retrieval |

Priority: frustration_recovery > testing_aware > meta_aware > repetition_aware > greeting > rapid_fire > exploratory > standard

### Conversation State

19-field state tracked per conversation:

- Tone tracking (formal/informal ratio)
- Repetition detection (embedding similarity history)
- Query pattern analysis (length, frequency, question ratio)
- **Precision mode** (auto-detected from query structure):
  - `concise` — short factual answers
  - `analytical` — structured comparisons
  - `speculative` — open-ended exploration
  - `implementation` — code and how-to
  - `adversarial` — challenge and critique

### 5 Precision Modes

| Mode | Detected By | Effect |
|------|------------|--------|
| `concise` | Short direct queries | Brief responses, minimal context |
| `analytical` | Comparison/evaluation queries | Structured, detailed |
| `speculative` | "What if" / exploratory queries | Exploratory tone |
| `implementation` | "How to" / code queries | Code-focused |
| `adversarial` | Challenge/critique queries | Critical analysis |

---

## Database Schema (9 Tables)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `conversations` | Conversation metadata + rolling topic embedding | id, title, message_count, topic_embedding (vector 768) |
| `chat_messages` | All user + assistant messages | conversation_id, role, content, timestamp |
| `user_queries` | Query embeddings for semantic Q&A search | query_text, embedding (vector 768), response_text |
| `user_profile` | Persistent user facts (key-value) | user_id, key, value, category |
| `document_chunks` | Knowledge base vectors | content, embedding (vector 768), source |
| `conversation_state` | Serialized ConversationState per conversation | state_data (JSONB) |
| `conversation_threads` | Topic thread centroids + metadata | centroid_embedding (vector 768), summary, label, message_count |
| `research_insights` | Extracted insights with embeddings | insight_type, insight_text, embedding (vector 768), confidence_score |
| `concept_links` | Cross-thread concept graph | concept, embedding (vector 768), source_type, source_id |

All tables managed by `query_db.py` (50+ functions). Single PostgreSQL + pgvector instance.

---

## API Surface (24 Routes)

### Core
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Batch response |
| `POST` | `/chat/stream` | SSE streaming (Vercel AI SDK compatible) |
| `POST` | `/chat/regenerate` | Re-generate last assistant turn |

### Conversations
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/conversations` | Create |
| `GET` | `/conversations` | List |
| `GET` | `/conversations/search` | Full-text search |
| `GET` | `/conversations/{id}` | Get by ID |
| `GET` | `/conversations/{id}/messages` | Messages |
| `PUT` | `/conversations/{id}` | Rename |
| `DELETE` | `/conversations/{id}` | Delete |
| `GET` | `/conversations/{id}/export` | JSON export |
| `GET` | `/conversations/{id}/state` | Behavioral state (debug) |

### Research
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/conversations/{id}/threads` | Topic threads |
| `GET` | `/conversations/{id}/threads/{tid}` | Thread detail + insights |
| `GET` | `/conversations/{id}/insights` | Conversation insights |
| `GET` | `/conversations/{id}/concepts` | Concept links |
| `GET` | `/insights/search?q=...&type=...` | Cross-thread insight search |
| `GET` | `/concepts/search?q=...` | Cross-thread concept search |

### Profile
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/profile` | Get entries |
| `POST` | `/profile` | Add/upsert |
| `PUT` | `/profile/{id}` | Update |
| `DELETE` | `/profile/{id}` | Delete |

### System
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status, DB, version, model |
| `GET` | `/` | Serve frontend |

---

## Stage Streaming Protocol

Real-time pipeline observability via Vercel AI SDK data stream format:

```
8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]
8:[{"stage":"threaded","thread_id":"...","label":"Caching Architecture"}]
8:[{"stage":"retrieved","retrieval_info":{"num_docs":4}}]
8:[{"stage":"generating"}]
0:"The"
0:" answer"
0:" is..."
8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...}}]
e:{"finishReason":"stop"}
d:{"finishReason":"stop"}
```

---

## Extension Hooks

4 injection points. No fork required.

| Hook | Signature | When |
|------|-----------|------|
| `@Hooks.before_generation` | `fn(pipeline_result) → pipeline_result` | After pipeline, before LLM |
| `@Hooks.after_generation` | `fn(response, pipeline_result) → str` | After LLM, before persist |
| `@Hooks.policy_override` | `fn(features, decision) → decision` | After policy resolve |
| `@Hooks.before_persist` | `fn(pipeline_result, response_text) → None` | Before DB writes |

---

## File Map

```
backend/
├── main.py                  # FastAPI app · 12-step pipeline · 24 routes
├── settings.py              # 56 fields (53 env-overridable)
├── policy.py                # ★ Deterministic retrieval gating
├── topic_threading.py       # ★ EMA centroid thread resolution
├── research_memory.py       # ★ Insight extraction + concept linking
├── thread_summarizer.py     # Progressive per-thread summarization
├── conversation_state.py    # ★ Multi-signal state + precision modes
├── behavior_engine.py       # ★ 8-mode behavioral router
├── context_manager.py       # Token-budget history trimming
├── query_db.py              # PostgreSQL + pgvector (50+ functions, 9 tables)
├── vector_store.py          # Document search (pgvector + numpy fallback)
├── embeddings.py            # BAAI/bge-base-en-v1.5 · 768-dim · local
├── hooks.py                 # 4 extension hook points
├── cache.py                 # Optional Redis (graceful no-op)
├── worker.py                # Bounded ThreadPoolExecutor
├── cli.py                   # init · ingest · dev · memory inspect · memory query
├── DOCS.md                  # Full implementation documentation
└── llm/
    ├── providers/           # cerebras · openai · anthropic
    ├── client.py            # Active-provider wrapper
    ├── classifier.py        # 5-intent classification
    ├── prompts.py           # All prompt templates (single source of truth)
    ├── prompt_orchestrator.py # ★ Policy-aware message assembly
    ├── generators.py        # Streaming + batch generation
    └── profile_detector.py  # Personal fact extraction

knowledge/                   # Drop .txt/.md → auto-indexed on startup
frontend/                    # React 18 · Vite · Tailwind · Vercel AI SDK
tests/                       # 297 tests · 13 files · pure unit · no DB/LLM calls
```

`★` = study these files — each demonstrates a named pattern

---

## Configuration

56 settings in `settings.py`, 53 env-overridable. Key knobs:

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_PROVIDER` | `cerebras` | Provider selection |
| `RETRIEVAL_K` | `4` | Docs per query |
| `MAX_HISTORY_TOKENS` | `8000` | History budget |
| `TOPIC_DECAY_ALPHA` | `0.2` | EMA centroid decay rate |
| `THREAD_ATTACH_THRESHOLD` | `0.55` | Min similarity to join thread |
| `THREAD_SUMMARY_INTERVAL` | `8` | Summarize every N messages |
| `RESEARCH_INSIGHTS_ENABLED` | `true` | Background insight extraction |
| `CONCEPT_LINKING_ENABLED` | `true` | Cross-thread concept linking |
| `BEHAVIOR_ENGINE_ENABLED` | `true` | Behavioral routing |

Full reference: [`.env.example`](../.env.example) | [`settings.py`](../backend/settings.py)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Pluggable: Cerebras, OpenAI, Anthropic (or any via `base_url`) |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim, local, no API key) |
| Vector Search | PostgreSQL + pgvector (HNSW index) |
| Database | PostgreSQL 16 + pgvector (single DB for everything) |
| Backend | FastAPI + Uvicorn (Python 3.12) |
| Frontend | React 18 + Vite + Tailwind + Vercel AI SDK + Zustand |
| Streaming | Vercel AI SDK data stream protocol over SSE |
| Cache | Optional Redis (graceful degradation) |
| Background | ThreadPoolExecutor (bounded, atexit cleanup) |
| CLI | argparse subcommands: init, ingest, dev, memory inspect, memory query |
| Deploy | Docker Compose (PostgreSQL + optional Redis + app) |

---

## Quick Start

```bash
git clone <repo-url> && cd Chatapp
python backend/cli.py init        # Create knowledge/, copy .env
# Edit .env → LLM_API_KEY=your-key
docker compose up postgres -d
python backend/cli.py dev         # → http://localhost:8000
```

Or: `docker compose up --build` for everything.

---

## CLI

```bash
python backend/cli.py init                                # Scaffold project
python backend/cli.py ingest [DIR]                        # Index knowledge base
python backend/cli.py dev [--host HOST] [--port PORT]     # Dev server
python backend/cli.py memory inspect                      # Full cognitive state
python backend/cli.py memory inspect -c CONVERSATION_ID   # Specific conversation
python backend/cli.py memory inspect --insights-only      # Just insights
python backend/cli.py memory query "your search text"     # Cross-thread search
python backend/cli.py memory query "X" --type decision    # Filter by type
python backend/cli.py memory query "X" -k 5              # Limit results
```
