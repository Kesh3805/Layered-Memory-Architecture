# Developer Guide

This guide explains the project from an engineer's point of view. For the product-facing interpretation, see [10_product_guide.md](./10_product_guide.md).

## What This Project Actually Is

Layered Memory Architecture (LMA) is a research-oriented conversational AI system that tries to do more than standard RAG.

Instead of treating every message as "retrieve docs, then answer," it adds:

- intent classification before retrieval
- policy-driven context injection
- per-conversation behavioral state
- topic threading inside a conversation
- research memory across turns
- instrumentation to measure whether those extra layers are worth the cost

In practical terms, the repo is:

- a [FastAPI backend](../backend/main.py) that runs the pipeline
- a [React + Vite frontend](../frontend/src/App.tsx) that exposes chat plus observability
- a [PostgreSQL + pgvector](../backend/query_db.py) persistence layer
- a [knowledge/](./) folder that doubles as the ingestible RAG corpus
- an [experiments/](../experiments/README.md) harness for A/B evaluation

## Current Reality At A Glance

Verified from the current codebase:

- 4 memory tiers are implemented
- 9 PostgreSQL tables back the system
- 32 FastAPI routes are defined in [backend/main.py](../backend/main.py)
- 361 tests are currently collected by `pytest --collect-only -q`
- the backend version is `6.0.0`, but a few peripheral files still show older version labels

This matters because some older docs in the repo are directionally right but slightly stale on counts, commands, and roadmap state.

## The Fast Mental Model

If you need one sentence:

> This is a policy-gated, stateful AI assistant that tries to make retrieval conditional, inspectable, and experimentally measurable.

If you need one diagram:

```text
Frontend UI
  -> /chat or /chat/stream
  -> run_pipeline()
     -> embed query
     -> load history + profile
     -> classify intent
     -> behavior engine
     -> topic threading
     -> research context
     -> policy resolve
     -> selective retrieval
     -> prompt assembly
     -> LLM generation
  -> background persistence
     -> save messages
     -> update profile
     -> update topic vectors
     -> extract insights
     -> link concepts
     -> summarize thread
```

## End-to-End Request Flow

The canonical flow lives in [backend/main.py](../backend/main.py), mostly inside `run_pipeline()` and `persist_after_response()`.

### 1. Request enters through chat

The frontend usually calls [`POST /chat/stream`](../backend/main.py), not the non-streaming route. The custom stream client is in [frontend/src/hooks/use-chat-stream.ts](../frontend/src/hooks/use-chat-stream.ts).

Important detail:

- the frontend does not use `useChat` directly
- it implements its own parser for the Vercel AI SDK stream protocol

### 2. Query embedding is generated locally

[backend/embeddings.py](../backend/embeddings.py) loads a sentence-transformers model lazily and runs embeddings locally. Default model:

- `BAAI/bge-base-en-v1.5`

This means retrieval quality depends on local embedding model choice, not only the LLM provider.

### 3. History and profile load in parallel

The backend parallelizes:

- query embedding
- recent message fetch
- profile fetch

This happens in a `ThreadPoolExecutor` inside [`run_pipeline()`](../backend/main.py).

### 4. Intent classification happens before retrieval

[backend/llm/classifier.py](../backend/llm/classifier.py) tries fast heuristics first:

- greeting
- profile statement
- privacy question
- short continuation

If those miss, it falls back to an LLM classifier prompt in [backend/llm/prompts.py](../backend/llm/prompts.py).

Why this matters:

- retrieval is not automatic
- many low-value queries can skip expensive context assembly

### 5. Continuation messages are topic-gated

If a message looks like a continuation, the backend checks cosine similarity against the conversation topic vector. If similarity is too low, it gets downgraded back to `general`.

This logic prevents "What about that?" from attaching to the wrong topic after a domain jump.

### 6. Conversational state and behavior are updated

[backend/conversation_state.py](../backend/conversation_state.py) tracks:

- tone
- repetition
- testing/meta behavior
- intent streaks
- short-query streaks
- precision mode

[backend/behavior_engine.py](../backend/behavior_engine.py) turns that into a `BehaviorDecision`, which can:

- skip retrieval
- reduce retrieval
- boost retrieval
- change personality framing
- change response length

This is one of the repo's core ideas: intent answers "what is this message," while behavior answers "what experience should the system deliver."

### 7. Topic threads are resolved

[backend/topic_threading.py](../backend/topic_threading.py) groups messages into per-conversation topical threads using centroid similarity.

Behavior:

- attach to nearest thread if similarity passes threshold
- otherwise create a new thread
- update centroid with EMA
- summarize every `THREAD_SUMMARY_INTERVAL`

### 8. Research memory is retrieved

[backend/research_memory.py](../backend/research_memory.py) pulls back:

- semantically related prior insights
- semantically related concept links

This is the project's most ambitious layer, and also the one whose ROI is currently the weakest in published results.

### 9. Policy decides what context gets injected

[backend/policy.py](../backend/policy.py) is the hard-routing layer.

It decides whether to inject:

- RAG documents
- prior Q&A
- profile context
- privacy framing
- curated history

This separation is important:

- `classifier.py` decides intent
- `behavior_engine.py` modulates experience
- `policy.py` decides retrieval and injection

### 10. Retrieval happens selectively

Document retrieval goes through [backend/vector_store.py](../backend/vector_store.py), which supports:

- pure pgvector cosine retrieval
- hybrid BM25 + vector fusion via [backend/hybrid_search.py](../backend/hybrid_search.py)
- optional cross-encoder reranking via [backend/reranker.py](../backend/reranker.py)

Q&A retrieval comes from:

- `user_queries` across conversations
- same-conversation semantic matches for continuity

### 11. Prompt assembly is modular

[backend/llm/prompt_orchestrator.py](../backend/llm/prompt_orchestrator.py) assembles the final message list from:

- system prompt
- greeting frame
- behavior frame
- thread frame
- research frame
- profile frame
- RAG frame
- Q&A frame
- conversation history
- latest user message

This is also where token budgeting and optional history summarization are applied.

### 12. Response is generated, then persistence happens in the background

Generation is in [backend/llm/generators.py](../backend/llm/generators.py). Background work is delegated to [backend/worker.py](../backend/worker.py), which currently uses an in-process bounded thread pool.

After each response, the system may:

- save messages and query embeddings
- detect and save profile updates
- update conversation topic vector
- auto-title the conversation
- extract research insights
- link concepts
- summarize threads

## The Four Memory Tiers

The repo's core concept is not "chat history with retrieval." It is explicit layered memory.

| Tier | Where it lives | What it stores | Why it exists |
|---|---|---|---|
| Episodic | `user_queries`, `chat_messages` | raw messages, Q&A, embeddings | semantic continuity and history search |
| Semantic profile | `user_profile` | user facts, preferences, identity | personalization and privacy transparency |
| Conversational state | `conversation_state` | tone, repetition, testing, precision mode | behavior adaptation |
| Research memory | `research_insights`, `concept_links` | decisions, hypotheses, concepts | long-horizon structured recall |

## Backend Module Map

If you are onboarding, these are the files that matter most.

| File | Why you read it first |
|---|---|
| [backend/main.py](../backend/main.py) | main request pipeline, routes, persistence orchestration |
| [backend/policy.py](../backend/policy.py) | deterministic routing logic |
| [backend/behavior_engine.py](../backend/behavior_engine.py) | experience-level modulation |
| [backend/conversation_state.py](../backend/conversation_state.py) | state model and detection heuristics |
| [backend/topic_threading.py](../backend/topic_threading.py) | thread attach/create logic |
| [backend/research_memory.py](../backend/research_memory.py) | insight extraction and concept linking |
| [backend/query_db.py](../backend/query_db.py) | schema, persistence, semantic search queries |
| [backend/vector_store.py](../backend/vector_store.py) | retrieval mode selection |
| [backend/llm/prompt_orchestrator.py](../backend/llm/prompt_orchestrator.py) | prompt construction |
| [backend/telemetry.py](../backend/telemetry.py) | evaluation-ready instrumentation |

### If You Want To Change X, Edit Y

| Change | Primary files |
|---|---|
| add or refine an intent | `backend/llm/classifier.py`, `backend/llm/prompts.py`, `backend/policy.py` |
| change retrieval gating | `backend/policy.py`, `backend/behavior_engine.py`, `backend/settings.py` |
| change prompt wording | `backend/llm/prompts.py` |
| change conversation personality behavior | `backend/behavior_engine.py`, `backend/conversation_state.py` |
| tune continuation sensitivity | `backend/settings.py`, `backend/main.py` |
| tune topic threads | `backend/topic_threading.py`, `backend/thread_summarizer.py` |
| tune research extraction | `backend/research_memory.py`, `backend/settings.py` |
| add persistence fields | `backend/query_db.py`, tests touching that module |
| add a new LLM provider | `backend/llm/providers/` and provider loader |
| add experiments | `experiments/`, `backend/telemetry.py`, `/experiments/config` |

## Frontend Architecture

The frontend is not just a chat shell. It is an inspection surface for the pipeline.

### Core pieces

- [frontend/src/App.tsx](../frontend/src/App.tsx): top-level shell
- [frontend/src/store.ts](../frontend/src/store.ts): global Zustand state
- [frontend/src/hooks/use-chat-stream.ts](../frontend/src/hooks/use-chat-stream.ts): custom streaming client
- [frontend/src/components/AIMessage.tsx](../frontend/src/components/AIMessage.tsx): message renderer plus memory/debug panels
- [frontend/src/components/ai/AIThreadPanel.tsx](../frontend/src/components/ai/AIThreadPanel.tsx): thread sidebar
- [frontend/src/components/ai/AIResearchDashboard.tsx](../frontend/src/components/ai/AIResearchDashboard.tsx): full research overlay
- [frontend/src/components/ProfileModal.tsx](../frontend/src/components/ProfileModal.tsx): profile CRUD UI

### Product-important frontend behaviors

- optimistic user/assistant message insertion during stream startup
- stage annotations rendered before tokens arrive
- per-message retrieval and debug panels
- auto-refresh of threads, insights, and concepts after each response

### One implementation detail to know

The frontend package depends on `ai`, but the chat flow uses a custom hook, not the full `useChat` abstraction. The app only borrows the stream protocol shape.

## Database And Persistence Model

[backend/query_db.py](../backend/query_db.py) creates and migrates the schema at app startup. There is no external migration tool; schema evolution is handled by `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... IF NOT EXISTS`.

### Tables

| Table | Purpose |
|---|---|
| `conversations` | top-level conversation metadata and rolling topic vector |
| `chat_messages` | full message transcript |
| `user_queries` | semantic Q&A history |
| `user_profile` | structured profile memory |
| `document_chunks` | knowledge-base vector store |
| `conversation_state` | persisted behavior state |
| `conversation_threads` | topic threads |
| `research_insights` | extracted structured insights |
| `concept_links` | linked concepts across threads |

### Important persistence design choices

- PostgreSQL + pgvector is the only real storage backend
- if the DB is unavailable, the app still runs in degraded mode
- degraded mode means no durable conversations, no profile memory, and no research/thread state
- document retrieval falls back to an in-memory vector list when the DB is unavailable

## Configuration Surface

The source of truth is [backend/settings.py](../backend/settings.py). It currently exposes 67 settings fields.

### Setting groups that matter most

- provider: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`
- retrieval: `RETRIEVAL_K`, `QA_K`, `SIMILARITY_THRESHOLD`
- hybrid and reranker: `HYBRID_*`, `RERANKER_*`
- behavior: `BEHAVIOR_ENGINE_ENABLED`, repetition thresholds, persistence flag
- threading: `THREAD_ENABLED`, attach threshold, summary interval
- research: `RESEARCH_INSIGHTS_ENABLED`, confidence floor, concept linking
- history budgeting: `MAX_CONTEXT_WINDOW`, `MAX_HISTORY_TOKENS`, summarization
- experiments: `BASELINE_MODE`

### Runtime toggles without restart

The experiment endpoints in [backend/main.py](../backend/main.py) can toggle these at runtime:

- behavior engine
- thread engine
- research insights
- concept linking
- hybrid search
- reranker

That is useful for A/B experiments and subsystem isolation.

## Local Development Workflows

## Backend

From the repo root, the safest commands are:

```bash
python backend/cli.py init
python backend/cli.py ingest
python backend/cli.py dev
```

Alternative after install:

```bash
rag-chat init
rag-chat ingest
rag-chat dev
```

Important note:

- some older docs in the repo say `python cli.py ...`
- from the repo root, the real file path is `backend/cli.py`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite runs on `5173` and proxies several backend routes to `8000` via [frontend/vite.config.ts](../frontend/vite.config.ts).

If you add new backend route families that the browser must call directly in dev, you may need to extend the proxy list.

## Docker

```bash
docker compose up --build
```

This starts:

- `postgres` with pgvector
- `app` with the built frontend and backend

Redis is optional and commented out by default.

## Testing And Evaluation

### Unit tests

```bash
pytest
```

Current test coverage is broad across:

- classifier behavior
- policy rules
- conversation state
- behavior engine
- topic threading
- research memory
- prompt assembly
- settings
- evaluation logic

### Experiments

Use the experiments harness when you want to know whether a feature earns its complexity.

Useful entry points:

- [experiments/README.md](../experiments/README.md)
- [backend/evaluation.py](../backend/evaluation.py)
- `POST /retrieval/test`
- `GET/POST /experiments/config`
- telemetry endpoints under `/telemetry`

## Observability Story

This repo is unusually strong on inspectability.

### Backend observability

[backend/telemetry.py](../backend/telemetry.py) records:

- intent source and confidence
- topic gate behavior
- behavior mode and triggers
- policy decisions
- thread routing
- retrieval counts and similarity metrics
- prompt token estimates
- per-stage latencies

### Frontend observability

The UI exposes:

- stage chips during streaming
- retrieval details per answer
- debug metadata per answer
- thread summaries and insights
- concept and research views

This is one of the clearest architectural strengths of the project.

## Known Rough Edges And Important Truths

These are worth knowing before you treat the repo like a finished platform.

### 1. It is effectively single-user right now

`user_id` exists in the backend schema and request models, but the frontend does not manage identity or auth. In practice, most flows run under `DEFAULT_USER_ID=public`.

### 2. Research memory is implemented, but current evidence says ROI is low

The published experiment notes in [README.md](../README.md) show threading and behavior have measurable value, but research memory currently has weak demonstrated payoff in the synthetic runs.

### 3. Hybrid search and reranking are not automatically wins

The retrieval experiment results in [README.md](../README.md) argue that, on the current small corpus size, hybrid and reranker modes add cost without improving retrieval quality enough to justify themselves.

### 4. There is some doc and UI drift

Examples verified in code:

- [frontend/src/components/ai/AIRetrievalPanel.tsx](../frontend/src/components/ai/AIRetrievalPanel.tsx) still says "retrieved via FAISS" even though the backend uses pgvector
- [start_server.bat](../start_server.bat) prints `v4.1.0` while the backend app declares `6.0.0`
- older knowledge docs still use shortened CLI commands that are ambiguous from repo root
- some README roadmap lines are stale relative to already-published retrieval findings

### 5. A few frontend API wrappers are stale or unused

Examples:

- `getThread()` in [frontend/src/api.ts](../frontend/src/api.ts) expects a `Thread`, but the backend returns `{thread, insights}`
- `searchConcepts()` expects `{concepts: ...}`, but the backend returns `{results: ...}`

These do not appear to be active bugs today because those wrappers are not used by the visible UI, but they are maintenance landmines.

### 6. Migrations are application-managed

There is no Alembic or dedicated migration system. Schema changes are encoded in `query_db.init_db()`. That is fast for research work, but risky for production hardening.

### 7. Background work is in-process

Insight extraction and thread summarization run via a local thread pool, not a durable task queue. That keeps setup simple, but limits production robustness.

## Best Reading Order For A New Engineer

If you want the fastest deep understanding, read in this order:

1. [README.md](../README.md)
2. [backend/main.py](../backend/main.py)
3. [backend/policy.py](../backend/policy.py)
4. [backend/conversation_state.py](../backend/conversation_state.py)
5. [backend/behavior_engine.py](../backend/behavior_engine.py)
6. [backend/topic_threading.py](../backend/topic_threading.py)
7. [backend/research_memory.py](../backend/research_memory.py)
8. [backend/query_db.py](../backend/query_db.py)
9. [frontend/src/hooks/use-chat-stream.ts](../frontend/src/hooks/use-chat-stream.ts)
10. [frontend/src/components/AIMessage.tsx](../frontend/src/components/AIMessage.tsx)

## If I Were Starting Work Here Tomorrow

I would keep these principles in mind:

- treat `policy.py` as the behavioral contract
- treat `main.py` as orchestration glue, not a place to add ad hoc rules
- measure subsystem value before making the pipeline more complex
- preserve inspectability whenever adding a new layer
- be careful when adding docs to `knowledge/`, because that changes the retrieval corpus

That last point is especially relevant now: these new guides live in `knowledge/`, so they will be ingestible by the app on the next index pass.
