<div align="center">

```
██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║
██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

### **The self-hosted research intelligence engine.**
*Not just another chatbot. A system that remembers, threads, and grows with your thinking.*

[![Version](https://img.shields.io/badge/version-6.0.0-blueviolet.svg?style=flat-square)](DOCS.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/tests-254%20passing-brightgreen.svg?style=flat-square)](#test-suite)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## The Problem With Every Other RAG System

Every RAG chatbot has the same invisible flaw: **it treats every conversation like a blank slate.**

Ask about a topic, get chunks. Ask a follow-up, get the same chunks again — or worse, completely different ones. The system never builds understanding. Every message is an isolated retrieval event. You end up doing all the intellectual work yourself, re-establishing context, re-summarizing where you left off, re-explaining what you already concluded.

**ResearchBot is built differently.**

It tracks which thread of thought you're in. It remembers what you've already decided. It knows when you're stress-testing an idea vs. asking for implementation steps. It links concepts across different conversations. It gets *smarter* the longer you use it.

---

## What It Actually Does

```
You ask a question
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  INTENT GATE                                              │
│  Classifies before retrieving — general / continuation /  │
│  knowledge_base / profile / privacy                       │
└─────────────────────┬─────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   TOPIC THREADING         │  ← Where does this message belong?
        │   EMA centroid similarity │    Attach to existing thread or start new one
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   RESEARCH CONTEXT        │  ← What do we already know about this?
        │   Semantic insight search │    Decisions, conclusions, open questions
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   PRECISION MODE          │  ← How should we respond?
        │   analytical / concise /  │    Detected from query structure
        │   speculative / impl /    │
        │   adversarial             │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   GENERATION              │  ← Full context: thread + insights +
        │   Stream or batch         │    concepts + policy-resolved documents
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   BACKGROUND PERSIST      │  ← Extract insights, link concepts,
        │   Research memory update  │    update thread centroid, summarize
        └───────────────────────────┘
```

---

## Four Layers of Memory

Most systems have one kind of memory: the last N messages. ResearchBot has four.

| Tier | Where | What It Stores | Lives |
|------|--------|----------------|-------|
| **A — Episodic** | `user_queries` table | Raw query embeddings, QA pairs | Permanent |
| **B — Semantic** | `user_profile` table | Who you are, what you do, your preferences | Permanent |
| **C — Conversational** | `conversation_state` table | Tone, repetition, patterns, precision mode | Per-conversation |
| **D — Research** | `research_insights` + `concept_links` | Decisions, conclusions, hypotheses, concept graphs | Permanent, cross-thread |

The research tier is what makes this different. After every response, the system runs a background extraction pass — pulling out decisions you've made, conclusions you've reached, open questions you haven't answered, and observations worth keeping. These are embedded and indexed. Next time you explore a related topic, they surface automatically.

---

## Precision Modes

The system detects *how* you're thinking and adapts its response style automatically — no slash commands, no manual mode switching.

| Mode | Triggered By | Response Style |
|------|-------------|----------------|
| `analytical` | Default research posture | Structured breakdown, trade-offs, numbered reasoning |
| `concise` | Rapid-fire messages, very short queries | Minimal prose, bullets, direct answer first |
| `speculative` | "what if", "hypothetically", "suppose" | Explores implications, labels assumptions vs. facts |
| `implementation` | "implement", "code", "build", "deploy" | Leads with runnable code, exact paths, gotchas |
| `adversarial` | "that's wrong", "but", "counterpoint" | Engages honestly with critique, defends with evidence |

---

## Topic Threading

Long research sessions fragment into incoherent noise in normal chatbots. ResearchBot groups your messages into **topical threads** using embedding centroid similarity.

- Each thread has a **progressive summary** — compressed at every 8-message milestone
- Threads get **auto-generated labels** ("Vector Search Architecture", "Caching Strategy Trade-offs")
- The active thread's summary and recent insights are injected into every prompt
- Up to 12 active threads per conversation, each independently tracked

```
Conversation
├── Thread: "Database Architecture" ──────── 14 msgs ── Summary: "Decided on pgvector..."
├── Thread: "Caching Strategy" ───────────── 6 msgs  ── Summary: "Redis vs in-memory..."
└── Thread: "API Design" (active) ─────────── 3 msgs  ── No summary yet
         ^
         You are here — this thread's context is in your prompt
```

---

## Why Self-Host?

| | ResearchBot | ChatGPT / Claude |
|--|------------|-----------------|
| **Your data** | Stays on your infra — always | Sent to their servers |
| **Knowledge base** | Your private documents via RAG | None (unless custom GPT) |
| **Memory control** | Full — read, edit, delete | Opaque |
| **Cost** | Inference cost only | Subscription + inference |
| **Audit trail** | Every routing decision visible | Black box |
| **Custom logic** | 4 hook points, no fork needed | Not possible |
| **LLM portability** | Swap provider in 2 env vars | Locked in |
| **Local/offline** | Ollama support built in | Not possible |

---

## Quick Start

**Prerequisites:** Docker Desktop, Python 3.12+, an LLM API key (or Ollama running locally).

```bash
git clone <repo-url> && cd Chatapp

# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your environment
cp .env.example .env
# Open .env — set LLM_API_KEY, LLM_PROVIDER, and POSTGRES_* vars

# 4. Start PostgreSQL with pgvector
docker compose up postgres -d

# 5. Start the server
python backend/cli.py dev     # → http://localhost:8000
```

**Docker (one command):**

```bash
docker compose up --build    # Starts PostgreSQL + app + auto-ingest
```

**Drop your documents in `knowledge/` and they auto-index on startup.**  
Supports `.txt` and `.md`. The system chunks, embeds (768-dim local model), and stores in pgvector — no configuration needed.

---

## LLM Providers

Swap your provider any time — no code changes, just environment variables.

```env
# Cerebras (default — fast inference)
LLM_PROVIDER=cerebras
LLM_API_KEY=csk-...

# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-20241022

# Fully local — no API key, no data leaves your machine
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

Add a new provider by subclassing `LLMProvider` in `llm/providers/base.py` — two methods required.

---

## API Reference

### Core

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Batch chat response |
| `POST` | `/chat/stream` | Streaming response (Vercel AI SDK SSE format) |
| `POST` | `/chat/regenerate` | Re-generate last assistant turn |

### Research Dashboard

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/conversations/{id}/threads` | All threads with labels + summaries |
| `GET` | `/conversations/{id}/threads/{tid}` | Thread detail — centroid, summary, message count |
| `GET` | `/conversations/{id}/insights` | Extracted decisions, conclusions, open questions |
| `GET` | `/conversations/{id}/concepts` | Concept links extracted from this conversation |
| `GET` | `/concepts/search?q=...` | Semantic search across all concept links |

### Conversations & Profile

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/conversations` | List all conversations |
| `GET` | `/conversations/search?q=` | Full-text search across messages |
| `GET` | `/conversations/{id}/export` | Export as JSON or plaintext |
| `GET/POST/PUT/DELETE` | `/profile` | Per-user profile management |
| `GET` | `/health` | Server status, DB state, version, model info |

---

## Extension Hooks

Four injection points let you customize behavior without touching core logic:

```python
from hooks import Hooks

# Inject dynamic context before generation
@Hooks.before_generation
def add_current_date(result):
    from datetime import date
    result.rag_context += f"\n\nToday's date: {date.today().isoformat()}"
    return result

# Override routing decisions
@Hooks.policy_override
def always_rag_for_questions(features, decision):
    if "?" in features.query:
        decision.inject_rag = True
    return decision

# Run after response is generated but before saving
@Hooks.after_generation
def log_response_length(response, result):
    print(f"[{result.intent}] {len(response)} chars")
    return response
```

Hook points: `before_generation` · `after_generation` · `policy_override` · `before_persist`

---

## Project Structure

```
backend/
├── main.py                  # FastAPI app · 12-step pipeline · all endpoints
├── settings.py              # ~53 settings, all env-overridable
├── policy.py                # Deterministic retrieval policy — auditable rules
├── topic_threading.py       # EMA centroid thread resolution          ← v6
├── research_memory.py       # Insight extraction + concept linking     ← v6
├── thread_summarizer.py     # Progressive per-thread summarization     ← v6
├── conversation_state.py    # 4-tier memory state + precision modes    ← v6
├── behavior_engine.py       # 8-mode behavioral router
├── context_manager.py       # Token-budget history trimming
├── query_db.py              # PostgreSQL + pgvector (50+ CRUD functions)
├── vector_store.py          # Document search (pgvector + numpy fallback)
├── embeddings.py            # BAAI/bge-base-en-v1.5 · 768-dim · local
├── hooks.py                 # 4 extension hook points
├── cache.py                 # Optional Redis (graceful no-op when disabled)
├── worker.py                # Bounded ThreadPoolExecutor background tasks
├── cli.py                   # init / ingest / dev commands
└── llm/
    ├── providers/           # cerebras · openai · anthropic · (add your own)
    ├── client.py            # Active-provider wrapper
    ├── classifier.py        # 5-intent classification (heuristics → LLM)
    ├── prompts.py           # All prompt templates (single source of truth)
    ├── prompt_orchestrator.py # Policy-aware message assembly
    ├── generators.py        # Streaming + batch generation
    └── profile_detector.py  # Personal fact extraction

knowledge/                   # Drop .txt / .md files here → auto-indexed
frontend/                    # React 18 · Vite · Tailwind · Vercel AI SDK
tests/                       # 254 tests · zero DB/LLM calls required
```

---

## Configuration Reference

All settings live in `backend/settings.py`, driven by environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `cerebras` | `cerebras` · `openai` · `anthropic` |
| `LLM_API_KEY` | — | **Required.** Your provider key |
| `LLM_MODEL` | *(provider default)* | Model name override |
| `LLM_BASE_URL` | — | Custom endpoint for Ollama, Azure, vLLM |
| `DATABASE_URL` | — | Full PostgreSQL DSN |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local sentence-transformer model |
| `RETRIEVAL_K` | `4` | Document chunks per knowledge query |
| `MAX_HISTORY_TOKENS` | `8000` | Token budget for conversation history |
| `THREAD_ENABLED` | `true` | Enable topic threading engine |
| `THREAD_ATTACH_THRESHOLD` | `0.55` | Min similarity to join an existing thread |
| `THREAD_SUMMARY_INTERVAL` | `8` | Summarize thread every N messages |
| `RESEARCH_INSIGHTS_ENABLED` | `true` | Enable background insight extraction |
| `CONCEPT_LINKING_ENABLED` | `true` | Enable cross-thread concept linking |
| `BEHAVIOR_ENGINE_ENABLED` | `true` | Enable behavioral routing |
| `ENABLE_CACHE` | `false` | Redis caching for embeddings + classifications |
| `FORCE_REINDEX` | `false` | Re-index knowledge base on every startup |

Full reference: [`.env.example`](.env.example)

---

## Frontend

React 18 + Vite + Tailwind with built-in observability:

- **Intent badge** — classification + confidence on every message
- **Pipeline timeline** — live stage chips: `Classified → Threaded → Retrieved → Generating`
- **Thread panel** — active thread label, summary, and message count
- **Retrieval panel** — expandable view of documents, insights, and Q&A used
- **Research dashboard** — browsable threads, extracted insights, concept graph
- **Debug drawer** — raw `PolicyDecision` + `ThreadResolution` + `BehaviorDecision` JSON
- **Command palette** — `Ctrl+K` quick navigation

```bash
cd frontend
npm install
npm run dev      # → http://localhost:5173
npm run build    # → frontend/dist/ (auto-served by FastAPI)
```

---

## Test Suite

**254 tests** · Pure unit tests · No database or LLM calls required · `pytest --tb=short -q`

| File | Coverage |
|------|----------|
| `test_chunker.py` | Text splitting — paragraphs, sentences, characters, overlap, edge cases |
| `test_classifier.py` | Intent classification — heuristic paths, LLM fallback, caching |
| `test_context_manager.py` | Token budgeting, history fitting, progressive summarization |
| `test_policy.py` | All 5 intents, structural follow-up score, feature extraction |
| `test_prompt_orchestrator.py` | Message assembly, precision modes, thread/research context injection |
| `test_settings.py` | Defaults, env overrides, bool parsing, immutability |
| `test_conversation_state.py` | Tone detection, repetition, patterns, precision mode computation |
| `test_behavior_engine.py` | 8 behavior modes, priority ordering, retrieval modulation |
| `test_topic_threading.py` | EMA centroid math, cosine similarity, thread resolution |
| `test_research_memory.py` | Concept extraction heuristics, insight JSON parsing |
| `test_thread_summarizer.py` | Summarization, label generation, interval-based scheduling |

```bash
cd <project-root>
python -m pytest backend/tests/ -v
```

---

## CLI

```bash
python backend/cli.py init             # Scaffold project — create knowledge/, copy .env
python backend/cli.py ingest           # Index knowledge/ into pgvector
python backend/cli.py ingest docs/     # Index from a custom directory
python backend/cli.py dev              # Start dev server with hot-reload
python backend/cli.py dev --port 9000  # Custom port
```

---

## Roadmap

The foundations are production-hardened. Planned additions:

- [ ] **File upload ingestion** — PDF, DOCX, CSV → chunk → index without any CLI
- [ ] **Structured tool calling** — `tools` field in `ChatRequest`, LLM-driven tool use + result injection
- [ ] **Web search route** — Brave/Tavily API as a retrievable policy context source
- [ ] **Voice mode** — Whisper STT in frontend; TTS via OpenAI or Kokoro
- [ ] **Collaborative sessions** — shared conversation threads with divergent branching
- [ ] **Research export** — export full thread summaries + insights as structured Markdown report

---

## License

[MIT](LICENSE) — use it, fork it, build on it.

---

<div align="center">

**Built for people who think seriously and want software that keeps up.**

*Self-hosted · Private · Extensible · Actually remembers things*

</div>














