<div align="center">

```
███████╗████████╗ █████╗ ████████╗███████╗███████╗██╗   ██╗██╗
██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝██║   ██║██║
███████╗   ██║   ███████║   ██║   █████╗  █████╗  ██║   ██║██║
╚════██║   ██║   ██╔══██║   ██║   ██╔══╝  ██╔══╝  ██║   ██║██║
███████║   ██║   ██║  ██║   ██║   ███████╗██║     ╚██████╔╝███████╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝      ╚═════╝ ╚══════╝
```

### What ChatGPT would look like if it actually remembered your thinking.

**A reference implementation of the Layered Memory Architecture (LMA) for LLM systems.**

Multi-tier memory · Topic threading · Insight extraction · Deterministic retrieval routing

[![v6.0.0](https://img.shields.io/badge/version-6.0.0-blueviolet.svg?style=flat-square)](backend/DOCS.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/tests-297%20passing-brightgreen.svg?style=flat-square)](#test-suite)
[![MIT](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

> **Anti-pattern:** Dumping the last 20 messages into the context window and calling it "memory."
>
> Every major LLM framework does this. It works for demos. It breaks at scale.
> This architecture replaces that pattern with structured, persistent, inspectable cognition.

---

## The Problem

Every production LLM system hits the same wall.

Messages are isolated retrieval events. Context is a sliding window. "Memory" means the last N messages that fit the token budget. There is no structured recall, no concept evolution across sessions, no way to inspect why the system retrieved what it did.

The industry fix: stuff more tokens into the context window and hope for the best.

**This is not memory. This is a buffer.**

---

## Layered Memory Architecture (LMA)

This repository codifies a named architecture for giving LLM systems persistent, structured, inspectable cognition. Not a framework. Not a wrapper. A **pattern** — implemented as a working system you can run, study, and extend.

```
1. Memory is structured, not appended.
2. Retrieval is policy-bound, not automatic.
3. Behavior is inferred, not manually toggled.
4. State is inspectable at every layer.
5. Determinism first. Generation second.
```

Every design decision in this repo flows from these five rules. If you read nothing else, read the [implementation of rule 2](backend/policy.py) and the [implementation of rule 1](backend/research_memory.py).

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
├── settings.py              # 53 settings, env-overridable, frozen dataclass
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
tests/                       # 297 tests · pure unit · no DB/LLM calls
```

`★` = study these files first

---

## Configuration

All 53 settings in [`backend/settings.py`](backend/settings.py), driven by env vars. Key knobs:

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

## Roadmap

- [ ] File upload ingestion — PDF, DOCX, CSV
- [ ] Structured tool calling — LLM-driven tool use + result injection
- [ ] Web search route — Brave/Tavily as a policy-gated context source
- [ ] Research export — thread summaries + insights as Markdown
- [ ] Collaborative sessions — shared threads with divergent branching

---

## License

[MIT](LICENSE)

---

<div align="center">

**This is how serious LLM systems should be built.**

*Study the patterns. Run the code. Build on the architecture.*

</div>














