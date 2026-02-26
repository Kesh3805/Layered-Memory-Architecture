<div align="center">

```
███████╗████████╗ █████╗ ████████╗███████╗███████╗██╗   ██╗██╗
██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝██║   ██║██║
███████╗   ██║   ███████║   ██║   █████╗  █████╗  ██║   ██║██║
╚════██║   ██║   ██╔══██║   ██║   ██╔══╝  ██╔══╝  ██║   ██║██║
███████║   ██║   ██║  ██║   ██║   ███████╗██║     ╚██████╔╝███████╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝      ╚═════╝ ╚══════╝
```

### A reference architecture for building stateful AI systems.

[![v6.0.0](https://img.shields.io/badge/version-6.0.0-blueviolet.svg?style=flat-square)](backend/DOCS.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/tests-254%20passing-brightgreen.svg?style=flat-square)](#test-suite)
[![MIT](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

> LLM systems are stateless by default.
> Human cognition is not.
> This repository implements the missing state layer.

---

## What This Is

The canonical implementation of **multi-tier memory**, **deterministic LLM routing**, and **inspectable AI pipelines** — packaged as a working system you can run, study, and extend.

Not a chatbot. Not a SaaS template. A **blueprint for how stateful AI systems should work**.

The patterns in this repo — policy-gated retrieval, centroid-based topic threading, background cognition extraction, precision-mode inference — are general. The chat interface is just the proof that they compose correctly.

---

## Design Principles

```
1. Memory is structured, not appended.
2. Retrieval is policy-bound, not automatic.
3. Behavior is inferred, not manually toggled.
4. State is inspectable, not opaque.
5. Determinism first. Generation second.
```

Every design decision flows from these five rules. If you read nothing else, read the [implementation of rule 2](backend/policy.py) and the [implementation of rule 1](backend/research_memory.py).

---

## What Breaks in Standard RAG

| Failure Mode | Why It Happens |
|-------------|---------------|
| **No thread continuity** | Every message is an isolated retrieval event. Follow-ups lose context. |
| **No persistent insight extraction** | The system never records what was decided, concluded, or left open. |
| **No cross-session concept linking** | Exploring "caching" in one thread and "Redis" in another — never connected. |
| **No reasoning-mode awareness** | "What if we used Kafka?" and "implement Kafka" trigger identical retrieval. |
| **No audit trail** | You can't inspect why the system retrieved what it did, or why it didn't. |
| **Retrieval on every query** | "Thanks" and "How do I deploy this?" both hit vector search. Wasteful and noisy. |

### What This Architecture Fixes

Every failure above maps to a specific subsystem:

| Failure | Fix | Implementation |
|---------|-----|----------------|
| No thread continuity | EMA centroid topic threading | [`topic_threading.py`](backend/topic_threading.py) |
| No insight extraction | Background LLM extraction pass | [`research_memory.py`](backend/research_memory.py) |
| No concept linking | Embedding-indexed concept graph | [`research_memory.py → link_concepts()`](backend/research_memory.py) |
| No reasoning-mode awareness | 5 precision modes, auto-detected | [`conversation_state.py`](backend/conversation_state.py) |
| No audit trail | Policy engine with full decision tracing | [`policy.py`](backend/policy.py) |
| Retrieval on every query | 5-intent classifier gates retrieval | [`llm/classifier.py`](backend/llm/classifier.py) |

---

## Architecture

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

## Four-Tier Memory Model

Most LLM systems have one tier of memory: the last N messages. This architecture has four.

| Tier | Storage | Contents | Lifespan |
|------|---------|----------|----------|
| **Episodic** | `user_queries` | Raw embeddings, QA pairs | Permanent |
| **Semantic** | `user_profile` | Identity, preferences, expertise | Permanent |
| **Conversational** | `conversation_state` | Tone, repetition patterns, precision mode | Per-conversation |
| **Research** | `research_insights` + `concept_links` | Decisions, conclusions, hypotheses, concept graphs | Permanent, cross-thread |

The research tier is the differentiator. After every response, a background pass extracts structured insights (decisions, conclusions, open questions, observations, hypotheses) and indexes them as embeddings. Next time a related topic surfaces — even in a different thread — those insights are retrieved and injected.

---

## Execution Trace

Real output shape from the pipeline. This is what the system computes before the LLM sees a single token.

```
─── Pipeline Trace ───────────────────────────────────────────────────

 Query:        "Should we use Redis or Memcached for the session layer?"
 Intent:       knowledge_base (confidence: 0.92)
 Thread:       "Caching Architecture" (similarity: 0.78, msg #6)
 Precision:    analytical

 Policy Decision:
   inject_rag:        true
   inject_history:    true
   inject_profile:    false
   skip_retrieval:    false
   retrieval_k:       4

 Thread Context:
   label:    "Caching Architecture"
   summary:  "Evaluating in-memory caching. Decided against local
              process cache due to horizontal scaling requirements.
              Redis preferred for pub/sub capability. Open question:
              session serialization format."
   msgs:     6

 Research Context:
   insights:
     [decision]       "Redis preferred for pub/sub capability" (0.82)
     [open_question]   "Session serialization format undecided" (0.71)
   concept_links:
     "Redis" ←→ "Caching Strategy" (0.89)
     "Horizontal Scaling" ←→ "Infrastructure" (0.74)

 Retrieved Docs:    4 chunks (max similarity: 0.81)
 History Budget:    2,847 / 8,000 tokens
 Background Jobs:   extract_insights, link_concepts, update_centroid

──────────────────────────────────────────────────────────────────────
```

Every field above is computable, traceable, and exposed via the API. Nothing is hidden.

---

## If You're Building With LLMs, Study These Files

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

## Memory Inspector

Inspect the full cognitive state from the command line:

```bash
python backend/cli.py memory inspect                     # All conversations
python backend/cli.py memory inspect --conversation ID   # Specific conversation
python backend/cli.py memory inspect --insights-only     # Just extracted insights
```

```
═══ Memory State ══════════════════════════════════════════════════

  Conversation: a1b2c3d4
  Threads: 3 active

  ┌─ Thread: "Database Architecture" ─────── 14 msgs ──────────┐
  │  Summary: Decided on pgvector over Pinecone. Single-DB      │
  │           architecture preferred. Open: index tuning.       │
  │  Insights:                                                  │
  │    [decision]       pgvector over Pinecone (0.91)           │
  │    [conclusion]     Single-DB reduces operational cost (0.85)│
  │    [open_question]  HNSW vs IVFFlat index strategy (0.78)   │
  └─────────────────────────────────────────────────────────────┘

  ┌─ Thread: "Caching Strategy" ──────────── 6 msgs ───────────┐
  │  Summary: Redis preferred for pub/sub. Memcached rejected.  │
  │  Insights:                                                  │
  │    [decision]       Redis over Memcached (0.88)             │
  │    [hypothesis]     Redis Cluster may be overkill (0.65)    │
  └─────────────────────────────────────────────────────────────┘

  Concept Links: 12
    "Redis" ←→ "Caching Strategy" (0.89)
    "pgvector" ←→ "Database Architecture" (0.92)
    ...

═══════════════════════════════════════════════════════════════════
```

### Cross-Thread Queries

Query across all threads and conversations from the CLI or the API:

```bash
python backend/cli.py memory query "unresolved questions about caching"
```

```
─── Cross-Thread Search ──────────────────────────────────────────

  Query: "unresolved questions about caching"

  Matching Insights:
    1. [open_question] "Session serialization format undecided"
       Thread: "Caching Strategy" │ Confidence: 0.78 │ Sim: 0.84

    2. [open_question] "Cache invalidation pattern for writes"
       Thread: "API Design" │ Confidence: 0.72 │ Sim: 0.71

  Matching Concepts:
    1. "Cache Invalidation" ←→ "API Design" (0.79)
    2. "Redis" ←→ "Caching Strategy" (0.89)

──────────────────────────────────────────────────────────────────
```

Also available via API:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/insights/search?q=...&type=open_question` | Cross-thread semantic insight search |
| `GET` | `/concepts/search?q=...` | Cross-thread concept search |

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

Swap with environment variables. No code changes.

```env
# Cerebras (default — fast inference)
LLM_PROVIDER=cerebras
LLM_API_KEY=csk-...

# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...

# Fully local via Ollama — no API key, nothing leaves your machine
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

Add providers by subclassing `LLMProvider` in [`llm/providers/base.py`](backend/llm/providers/base.py).

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

---

## Extension Hooks

Four injection points. No fork required.

```python
from hooks import Hooks

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
├── settings.py              # ~53 settings, env-overridable, frozen dataclass
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
    ├── classifier.py        # 5-intent classification
    ├── prompts.py           # All prompt templates
    ├── prompt_orchestrator.py # ★ Policy-aware message assembly
    ├── generators.py        # Streaming + batch generation
    └── profile_detector.py  # Personal fact extraction

knowledge/                   # Drop .txt/.md → auto-indexed
frontend/                    # React 18 · Vite · Tailwind · Vercel AI SDK
tests/                       # 254 tests · pure unit · no DB/LLM calls
```

`★` = study these files

---

## Configuration

All settings in [`backend/settings.py`](backend/settings.py), driven by env vars.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `cerebras` | `cerebras` · `openai` · `anthropic` |
| `LLM_API_KEY` | — | Provider key |
| `LLM_MODEL` | *(provider default)* | Model override |
| `LLM_BASE_URL` | — | Ollama / Azure / vLLM endpoint |
| `DATABASE_URL` | — | PostgreSQL DSN |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local embedding model |
| `RETRIEVAL_K` | `4` | Document chunks per query |
| `MAX_HISTORY_TOKENS` | `8000` | History token budget |
| `THREAD_ENABLED` | `true` | Topic threading |
| `THREAD_ATTACH_THRESHOLD` | `0.55` | Min similarity to join thread |
| `THREAD_SUMMARY_INTERVAL` | `8` | Summarize every N messages |
| `RESEARCH_INSIGHTS_ENABLED` | `true` | Background insight extraction |
| `CONCEPT_LINKING_ENABLED` | `true` | Cross-thread concept linking |
| `BEHAVIOR_ENGINE_ENABLED` | `true` | Behavioral routing |

Full reference: [`.env.example`](.env.example)

---

## Frontend

React 18 + Vite + Tailwind. Built for observability.

- **Pipeline timeline** — live stage chips: `Classified → Threaded → Retrieved → Generating`
- **Thread panel** — active thread label, summary, insights
- **Intent badge** — classification + confidence on every message
- **Research dashboard** — threads, insights, concept graph
- **Debug drawer** — raw `PolicyDecision` + `ThreadResolution` JSON
- **Command palette** — `Ctrl+K`

```bash
cd frontend && npm install && npm run dev    # → localhost:5173
```

---

## Test Suite

**254 tests** · Pure unit tests · Zero database or LLM calls · ~1 second

```bash
python -m pytest backend/tests/ -v
```

| File | Tests |
|------|-------|
| `test_policy.py` | Intent routing, structural follow-up, feature extraction |
| `test_topic_threading.py` | EMA centroid math, cosine similarity, thread resolution |
| `test_research_memory.py` | Concept extraction, insight JSON parsing |
| `test_thread_summarizer.py` | Summarization, labels, interval scheduling |
| `test_prompt_orchestrator.py` | Message assembly, precision/thread/research injection |
| `test_conversation_state.py` | Tone, repetition, precision mode computation |
| `test_behavior_engine.py` | 8 modes, priority, retrieval modulation |
| `test_classifier.py` | Heuristic paths, LLM fallback, caching |
| `test_context_manager.py` | Token budgeting, history fitting |
| `test_settings.py` | Defaults, env overrides, immutability |
| `test_chunker.py` | Paragraphs, sentences, overlap, edge cases |

---

## CLI

```bash
python backend/cli.py init                        # Scaffold project
python backend/cli.py ingest [DIR]                # Index knowledge base
python backend/cli.py dev [--port 9000]           # Dev server
python backend/cli.py memory inspect              # Full cognitive state
python backend/cli.py memory query "your search"  # Cross-thread semantic search
```

---

## Roadmap

- [ ] **File upload ingestion** — PDF, DOCX, CSV → chunk → index
- [ ] **Structured tool calling** — LLM-driven tool use + result injection
- [ ] **Web search route** — Brave/Tavily as a policy-gated context source
- [ ] **Voice mode** — Whisper STT / TTS
- [ ] **Research export** — thread summaries + insights as Markdown
- [ ] **Collaborative sessions** — shared threads with divergent branching

---

## License

[MIT](LICENSE)

---

<div align="center">

**A reference architecture for stateful AI.**

*Study it. Run it. Build on it.*

</div>














