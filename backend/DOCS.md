# Backend Implementation Documentation

> **Version 5.0.0** — Behavior-aware policy-driven intent-gated RAG chat framework

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Module Reference](#module-reference)
   - [main.py — Application & Pipeline](#mainpy--application--pipeline)
   - [settings.py — Configuration](#settingspy--configuration)
   - [policy.py — Behavior Policy Engine](#policypy--behavior-policy-engine)
   - [context_manager.py — Token Budgeting](#context_managerpy--token-budgeting)
   - [query_db.py — Persistence Layer](#query_dbpy--persistence-layer)
   - [vector_store.py — Document Index](#vector_storepy--document-index)
   - [embeddings.py — Local Embedding Model](#embeddingspy--local-embedding-model)
   - [chunker.py — Text Splitting](#chunkerpy--text-splitting)
   - [hooks.py — Extension Points](#hookspy--extension-points)
   - [cache.py — Optional Redis](#cachepy--optional-redis)
   - [worker.py — Background Tasks](#workerpy--background-tasks)
   - [cli.py — Command Line Interface](#clipy--command-line-interface)
   - [conversation_state.py — Conversational State](#conversation_statepy--conversational-state)
   - [behavior_engine.py — Behavioral Routing](#behavior_enginepy--behavioral-routing)
   - [llm/ Package](#llm-package)
3. [Pipeline Deep Dive](#pipeline-deep-dive)
4. [API Reference](#api-reference)
5. [Database Schema](#database-schema)
6. [Configuration Reference](#configuration-reference)
7. [Test Suite](#test-suite)
8. [Deployment](#deployment)

---

## Project Structure

```
Chatapp/
├── backend/                        # All Python server code
│   ├── __init__.py                 # Package marker
│   ├── main.py                     # FastAPI app + pipeline + all endpoints (~770 lines)
│   ├── settings.py                 # Centralized env-driven config (~45 settings)
│   ├── conversation_state.py       # Per-conversation behavioral state tracking
│   ├── behavior_engine.py          # Behavioral routing layer (8 modes)
│   ├── policy.py                   # BehaviorPolicy engine (deterministic rules)
│   ├── context_manager.py          # Token budgeting, history fitting, progressive summarization
│   ├── query_db.py                 # PostgreSQL + pgvector persistence (1012 lines, 31 functions)
│   ├── vector_store.py             # Document search (pgvector + in-memory fallback)
│   ├── embeddings.py               # BAAI/bge-base-en-v1.5 local embeddings (768-dim)
│   ├── chunker.py                  # Paragraph → sentence → character text splitting
│   ├── hooks.py                    # 4 decorator-based extension points
│   ├── cache.py                    # Optional Redis (no-op when disabled)
│   ├── worker.py                   # Bounded ThreadPoolExecutor for background tasks
│   ├── cli.py                      # CLI: init, ingest, dev commands
│   ├── .env.example                # Template with all settings documented
│   ├── llm/                        # LLM abstraction layer
│   │   ├── __init__.py             # Re-exports for backward compat
│   │   ├── client.py               # Thin wrapper delegating to active provider
│   │   ├── classifier.py           # 5-intent classification (heuristics + LLM)
│   │   ├── generators.py           # Response generation (stream + batch + titles)
│   │   ├── prompts.py              # ALL prompt templates (single source of truth)
│   │   ├── prompt_orchestrator.py  # Builds message lists from PolicyDecision
│   │   ├── profile_detector.py     # Extract personal info from user messages
│   │   └── providers/              # Pluggable LLM providers
│   │       ├── __init__.py         # Dynamic loader (reads LLM_PROVIDER from settings)
│   │       ├── base.py             # LLMProvider ABC (2 methods: complete, stream_text_deltas)
│   │       ├── cerebras.py         # Cerebras Cloud SDK
│   │       ├── openai.py           # OpenAI (also Azure, vLLM, Ollama via base_url)
│   │       └── anthropic.py        # Anthropic Messages API
│   └── tests/                      # 190+ unit tests
│       ├── conftest.py             # sys.path setup for flat imports
│       ├── __init__.py
│       ├── test_chunker.py         # 27 tests — chunk_text edge cases
│       ├── test_classifier.py      # Intent classification with mocked LLM
│       ├── test_context_manager.py # Token budgeting + summarization
│       ├── test_policy.py          # BehaviorPolicy rule coverage
│       ├── test_prompt_orchestrator.py # Message assembly + behavior frame verification
│       ├── test_settings.py        # Config defaults + env overrides
│       ├── test_conversation_state.py # Conversation state tracking (30 tests)
│       └── test_behavior_engine.py # Behavioral routing engine (24 tests)
├── frontend/                       # React 18 + Vite + Tailwind + AI SDK
│   └── src/
│       ├── components/ai/          # AI-native observability primitives
│       └── ...
├── knowledge/                      # Drop .txt/.md files here → auto-indexed
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml              # PostgreSQL + optional Redis + app
├── pyproject.toml                  # Build config + test config + ruff
├── requirements.txt                # Pinned dependencies
├── start_server.bat                # Windows one-click starter
├── start_server.sh                 # Linux/macOS one-click starter
├── ARCHITECTURE.md                 # High-level technical overview
└── README.md                       # Setup guide + feature overview
```

---

## Module Reference

### main.py — Application & Pipeline

**Purpose:** FastAPI application, the behavior-aware pipeline, all HTTP endpoints, and the UI serve layer.

**Key components:**

| Component | Description |
|---|---|
| `lifespan()` | Async context manager — init DB, vector store, auto-ingest knowledge on startup; drain worker on shutdown |
| `ChatRequest` | Pydantic model: `user_query`, `conversation_id?`, `tags?`, `user_id` |
| `RenameRequest` | Pydantic model: `title` |
| `NewConversationRequest` | Pydantic model: `title` (default "New Chat") |
| `ProfileEntryRequest` | Pydantic model: `key`, `value`, `category`, `user_id` |
| `RegenerateRequest` | Pydantic model: `conversation_id`, `user_id` |
| `PipelineResult` | Dataclass holding all pipeline outputs for generation |
| `run_pipeline(request)` | The shared pipeline — returns `PipelineResult` |
| `persist_after_response(p, response)` | Background persistence: save messages, detect profile, update topic vector, auto-title |
| `_ingest_knowledge()` | Reads `knowledge/` directory, chunks files, indexes into vector store |

**Pipeline steps (inside `run_pipeline`):**

| Step | What happens |
|---|---|
| 1 | Embed query via `get_query_embedding()` |
| 2 | Load history + profile from PostgreSQL (parallel with step 1 using `ThreadPoolExecutor(3)`) |
| 3 | Classify intent (pre-heuristics → LLM fallback → cache) |
| 4 | Topic gate: cosine similarity check for `continuation` — demotes to `general` if topic drift detected |
| 4b | Behavior engine: load/update conversational state, run behavioral routing (8 modes) |
| 5 | Extract `ContextFeatures` + `BehaviorPolicy.resolve()` + behavior overrides + `Hooks.run_policy_override()` |
| 6 | History pruning: recency window + semantic retrieval of relevant older Q&A |
| 7 | Selective context assembly: only inject what `PolicyDecision` flags say to inject |
| 8 | `Hooks.run_before_generation()` |
| 9 | Generate response (stream or batch via current LLM provider) |
| 10 | `Hooks.run_after_generation()` + `Hooks.run_before_persist()` |
| 11 | Background persist via `worker.submit()`: save messages, detect profile updates, update topic vector, auto-title first message |

**Concurrency optimization:** Steps 1–2 execute in a `ThreadPoolExecutor(max_workers=3)` to parallelize embedding, history fetch, and profile fetch (~90ms serial → ~50ms parallel).

---

### settings.py — Configuration

**Purpose:** Single source of truth for every tunable in the system. All values are environment-variable driven, frozen at startup.

**Architecture:**
- Uses `@dataclass(frozen=True)` for immutability
- `load_dotenv()` loads `.env` from project root (one level above `backend/`)
- Helper functions `_env()`, `_env_int()`, `_env_bool()`, `_env_float()` parse env vars with defaults

**Key setting groups:**

| Group | Settings | Defaults |
|---|---|---|
| **LLM** | `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | `cerebras`, `""`, `""`, `""` |
| **Token Budgets** | `MAX_CONTEXT_WINDOW`, `MAX_RESPONSE_TOKENS`, `MAX_HISTORY_TOKENS` | 65536, 2048, 8000 |
| **Classifier Tokens** | `MAX_CLASSIFIER_TOKENS`, `MAX_PROFILE_DETECT_TOKENS`, `MAX_TITLE_TOKENS` | 60, 200, 30 |
| **Embedding** | `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `QUERY_INSTRUCTION` | `BAAI/bge-base-en-v1.5`, 768, `"Represent this sentence..."` |
| **Retrieval** | `RETRIEVAL_K`, `QA_K`, `SIMILARITY_THRESHOLD` | 4, 4, 0.3 |
| **Continuity** | `TOPIC_CONTINUATION_THRESHOLD`, `RECENCY_WINDOW`, `SEMANTIC_K` | 0.35, 6, 3 |
| **Knowledge Base** | `KNOWLEDGE_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `FORCE_REINDEX` | `<project_root>/knowledge`, 500, 50, False |
| **Server** | `HOST`, `PORT`, `ALLOWED_ORIGINS` | `0.0.0.0`, 8000, `*` |
| **Database** | `DATABASE_URL`, `POSTGRES_*`, `DB_POOL_MIN`, `DB_POOL_MAX` | `""`, default creds, 2, 10 |
| **Cache** | `ENABLE_CACHE`, `REDIS_URL`, `CACHE_TTL` | False, `redis://localhost:6379/0`, 3600 |
| **Pipeline** | `DEFAULT_USER_ID`, `HISTORY_FETCH_LIMIT`, `ENABLE_HISTORY_SUMMARIZATION` | `default`, 50, True |
| **Behavior Engine** | `BEHAVIOR_ENGINE_ENABLED`, `BEHAVIOR_REPETITION_THRESHOLD`, `BEHAVIOR_PATTERN_WINDOW`, `BEHAVIOR_STATE_PERSIST` | True, 0.7, 10, True |

---

### policy.py — Behavior Policy Engine

**Purpose:** Deterministic rules that decide what context to inject into the LLM. Separates behavior from model-calling code — when behavior is wrong, you fix a rule here, never edit prompts.

**Components:**

#### Signal lists (shared with classifier)
- `GREETING_PATTERNS` — 17 patterns: "hello", "hi", "hey", "good morning", etc.
- `PERSONAL_REF_SIGNALS` — 32 phrases: "my job", "my name", "what do i do", etc.
- `PROFILE_STATEMENT_PREFIXES` — 20 prefixes: "i am ", "my name is ", "call me ", etc.

#### `ContextFeatures` dataclass
Computed features for the current message — input to policy rules:
- `is_greeting`, `references_profile`, `privacy_signal`
- `is_followup`, `is_profile_statement`, `is_profile_question`
- `topic_similarity`, `has_profile_data`, `profile_name`
- `conversation_length`

#### `extract_context_features(query, intent, profile_entries, ...)`
Computes features from current state:
1. Greeting detection — pattern match (≤8 words)
2. Personal reference detection — substring match
3. Profile statement vs question — prefix match + `?` absence
4. Profile name extraction — scans for `name`/`first_name`/`full_name`/`username` keys

#### `PolicyDecision` dataclass
What the pipeline should do — every field is a directive:
- `inject_profile` — include user profile in context
- `inject_rag` — fetch documents from pgvector
- `inject_qa_history` — include prior Q&A
- `use_curated_history` — use history (recency + semantic blend)
- `privacy_mode` — activate transparency rules
- `greeting_name` — personalize response with user's name
- `retrieval_route` — label for observability
- `rag_k`, `rag_min_similarity` — retrieval depth + relevance floor
- `qa_k`, `qa_min_similarity` — Q&A retrieval parameters

#### `BehaviorPolicy.resolve(features, intent)`
Intent-based rules:

| Intent | RAG | Profile | QA History | Privacy | Route |
|---|---|---|---|---|---|
| `privacy` | No | If has data | No | Yes | `privacy` |
| `profile` (statement) | No | No | No | No | `profile_update` |
| `profile` (question) | No | If has data | No | No | `profile` |
| `knowledge_base` | Yes | No | Yes | No | `rag` |
| `continuation` | Yes (min_sim=0.35) | No | Yes | No | `conversation` |
| `general` | Yes (min_sim=0.45) | No | No | No | `adaptive` |

**Cross-intent overlays:**
1. If user name is known and profile not already injected → set `greeting_name`
2. If message references profile data and profile exists → force `inject_profile=True`, clear greeting_name

---

### context_manager.py — Token Budgeting

**Purpose:** Enforce LLM context window limits. Count-based heuristics (last N messages) are insufficient — one long message can consume as many tokens as 20 short ones. This module enforces precision.

**Token estimation:** `len(text) // 4` — ~97% accurate for English prose, ~90% for code.

**Public API:**

| Function | Signature | Purpose |
|---|---|---|
| `estimate_tokens(text)` | `str → int` | Character-based token estimation |
| `message_tokens(msg)` | `dict → int` | Token cost of a single message (content + 10 overhead) |
| `history_tokens(messages)` | `list[dict] → int` | Total tokens for message list |
| `compute_history_budget(context_window, response_reserve, preamble_tokens, min_budget)` | `→ int` | Remaining budget after preamble + response reserve |
| `fit_messages_to_budget(messages, budget_tokens, min_recent)` | `→ list[dict]` | O(n) prefix-sum trimming, preserves last `min_recent` (default 4) |
| `summarize_old_turns(messages, max_history_tokens, completion_fn, min_recent)` | `→ list[dict]` | Progressive summarization — compresses overflow into LLM summary |

**Progressive summarization flow:**
1. If history fits budget → return unchanged
2. Split into overflow + tail (last `min_recent=6` messages)
3. Check for existing summary message in history
4. Build transcript (prior summary + overflow messages, capped at 4000 tokens)
5. LLM generates 3-8 sentence summary preserving: topics, decisions, personal facts, unresolved questions, technical details
6. Return `[summary_msg] + recent_messages`
7. On failure → fallback to `fit_messages_to_budget()`

---

### query_db.py — Persistence Layer

**Purpose:** PostgreSQL + pgvector handles ALL storage needs in a single database. No FAISS, no separate vector DB.

**Connection management:**
- `SimpleConnectionPool(min=2, max=10)` — psycopg2 connection pooling
- `get_connection()` / `put_connection(conn)` — pooled checkout/return
- `put_connection` defensively rolls back dirty connections before returning to pool

**Tables:**

| Table | Purpose | Key columns |
|---|---|---|
| `conversations` | Conversation metadata | `id`, `title`, `user_id`, `tags[]`, `topic_embedding`, `created_at`, `updated_at` |
| `chat_messages` | Individual messages | `id`, `conversation_id`, `role`, `content`, `timestamp` |
| `user_queries` | Per-query embeddings | `id`, `conversation_id`, `query_text`, `ai_response`, `embedding` (768-dim vector), `tags[]`, `timestamp` |
| `user_profile` | Key-value personal data | `id`, `user_id`, `key`, `value`, `category`, `updated_at` |
| `document_chunks` | Knowledge base vectors | `id`, `content`, `embedding` (768-dim vector), `source`, `created_at` |

**All vector columns use pgvector's `vector(768)` type with HNSW indexing.**

**Public functions (31 total):**

*Schema:*
- `init_db()` → Create all tables + extensions + indexes (autocommit mode for safe DDL)

*Connections:*
- `get_connection()`, `put_connection(conn)`

*Conversations (8):*
- `create_conversation(title, user_id, tags)` → UUID
- `get_all_conversations(user_id)` → list[dict]
- `get_conversation_messages(cid, limit)` → list[dict]
- `get_recent_chat_messages(cid, limit)` → list[dict] (for pipeline)
- `rename_conversation(cid, title)` → bool
- `delete_conversation(cid)` → bool
- `search_conversations(user_id, query, limit)` → list[dict] (ILIKE search)
- `export_conversation(cid, format)` → dict (JSON/text export)

*Messages (2):*
- `save_chat_message(cid, role, content)` → id
- `delete_last_assistant_message(cid)` → bool (for regenerate)

*Queries / Q&A (4):*
- `save_user_query(cid, query, response, embedding, tags)` → id
- `search_similar_queries(embedding, k, min_similarity)` → list (cross-conversation)
- `search_same_conversation_qa(cid, embedding, k, min_similarity)` → list
- `infer_tags(query)` → list[str]

*Profile (4):*
- `get_user_profile(user_id)` → list[dict]
- `update_profile_entry(key, value, category, user_id)` → id (UPSERT)
- `delete_profile_entry(entry_id)` → bool
- `batch_update_profile(entries, user_id)` → int

*Document chunks (4):*
- `store_document_chunks(chunks, source)` → int
- `search_document_chunks(embedding, k, min_similarity)` → list[str]
- `count_document_chunks()` → int
- `clear_document_chunks(source?)` → None

*Topic vectors (2):*
- `get_topic_vector(cid)` → ndarray | None
- `update_topic_vector(cid, new_embedding)` → exp. moving avg (α=0.3)

---

### vector_store.py — Document Index

**Purpose:** Thin abstraction over document storage. pgvector when DB is available, numpy cosine in-memory fallback when it's not.

| Function | Description |
|---|---|
| `init(db_enabled)` | Set storage backend at startup |
| `add_documents(chunks, source)` | Index document chunks (pgvector or in-memory) |
| `search(query, k?, min_similarity)` | Semantic search over indexed docs |
| `has_documents()` | Check if any documents exist |
| `count()` | Number of indexed chunks |
| `clear(source?)` | Remove indexed documents |

**Fallback mode:** When PostgreSQL is unavailable, embeddings are stored in a Python list and searched via numpy dot product + norm. Non-persistent — lost on restart.

---

### embeddings.py — Local Embedding Model

**Purpose:** Local sentence-transformer embedding — no external API key required.

**Default model:** `BAAI/bge-base-en-v1.5` — 768 dimensions, top MTEB ranking, optimized for asymmetric retrieval.

**Asymmetric retrieval pattern:**
- **Documents** encoded with `get_embedding()` / `get_embeddings()` — no prefix
- **Queries** encoded with `get_query_embedding()` — applies `QUERY_INSTRUCTION` prefix when set

| Function | Purpose |
|---|---|
| `get_embedding(text)` | Encode a document passage → `np.ndarray` (float32) |
| `get_query_embedding(text)` | Encode a search query with instruction prefix → `np.ndarray` |
| `get_embeddings(texts)` | Batch encode documents → `np.ndarray` |
| `get_dim()` | Return embedding dimension of loaded model |

**Lazy loading:** Model loaded on first call, cached as module-level singleton. ~1-2s warm-up, then near-instant.

---

### chunker.py — Text Splitting

**Purpose:** Semantic-aware text splitting. Single source of truth used by `main.py` (startup auto-indexing) and `cli.py` (manual ingest).

**Strategy (3-level hierarchy):**
1. **Paragraph split** — split on `\n\n+` to get natural paragraphs
2. **Sentence split** — paragraphs larger than `chunk_size` are split on `[.!?]\s+`
3. **Character window** — sentences larger than `chunk_size` use sliding window (last resort)
4. **Merge** — consecutive small atoms merged up to `chunk_size` to avoid tiny fragments

**Signature:** `chunk_text(text, chunk_size=500, chunk_overlap=50) → list[str]`

The overlap parameter controls how many characters from the end of one chunk appear at the start of the next, preventing information loss at chunk boundaries.

---

### hooks.py — Extension Points

**Purpose:** Customize pipeline behavior without modifying core code. Four decorator-based hooks run in registration order.

| Hook | Signature | When |
|---|---|---|
| `@Hooks.before_generation` | `fn(pipeline_result) → pipeline_result` | After retrieval, before LLM call |
| `@Hooks.after_generation` | `fn(response: str, pipeline_result) → str` | After LLM response, before persist |
| `@Hooks.policy_override` | `fn(features, decision) → decision` | After `BehaviorPolicy.resolve()` |
| `@Hooks.before_persist` | `fn(pipeline_result, response_text) → None` | Before DB writes |

**Usage example:**
```python
from hooks import Hooks

@Hooks.policy_override
def force_rag_for_questions(features, decision):
    if "?" in features.query:
        decision.inject_rag = True
    return decision
```

`Hooks.clear()` removes all registered hooks (used in tests).

---

### cache.py — Optional Redis

**Purpose:** Cache intent classifications (30 min TTL) and embeddings to avoid redundant computation.

**Design:** Full no-op when Redis is unavailable — zero impact on functionality.

| Function | Purpose |
|---|---|
| `get(key)` / `put(key, value, ttl?)` | Generic get/set |
| `get_classification(query)` / `set_classification(query, result)` | Intent cache (30 min TTL) |
| `get_embedding(text)` / `set_embedding(text, vector)` | Embedding cache |

**Activation:** Set `ENABLE_CACHE=true` + `REDIS_URL` in `.env`. Lazy initialization on first access.

---

### worker.py — Background Tasks

**Purpose:** Fire-and-forget background execution using a bounded `ThreadPoolExecutor(max_workers=4)`.

| Function | Purpose |
|---|---|
| `submit(fn, *args, **kwargs)` | Submit task to thread pool. Exceptions are logged, never raised. |
| `shutdown(wait=True)` | Drain the pool. Called by lifespan + `atexit.register`. |

---

### cli.py — Command Line Interface

**Purpose:** Three commands for project management.

| Command | Function | Purpose |
|---|---|---|
| `python backend/cli.py init` | `cmd_init` | Create `knowledge/` dir, copy `.env.example` → `.env` |
| `python backend/cli.py ingest [DIR]` | `cmd_ingest` | Index knowledge base files into pgvector |
| `python backend/cli.py dev [--host] [--port]` | `cmd_dev` | Start uvicorn dev server with hot-reload |

`cmd_dev` sets `cwd=backend/` so flat module imports work correctly.

---

### conversation_state.py — Conversational State

**Purpose:** The "C tier" memory layer — per-conversation behavioral state tracking. Complements episodic memory (user_queries) and semantic memory (user_profile) with a meta-conversational layer that tracks *how* the user interacts, not *what* they said.

**Memory tiers:**
| Tier | Module | What it stores |
|---|---|---|
| A) Episodic | user_queries table | Facts extracted from queries |
| B) Semantic | user_profile table | User traits and preferences |
| C) Conversational | conversation_state | Behavioral patterns per conversation |

**`ConversationState` dataclass** (19 fields):
- **Topic:** `current_topic`, `topic_turns_stable`, `topic_drift_count`
- **Tone:** `emotional_tone` (neutral/positive/frustrated/curious/playful), `tone_shift_count`
- **Behavior:** `interaction_pattern`, `testing_flag`, `repetition_count`, `meta_comment_count`
- **Intent history:** `last_intent`, `intent_history` (deque), `intent_streak`
- **Stats:** `message_count`, `avg_query_length`, `short_query_streak`
- **Personality:** `dynamic_personality_mode`
- **Timing:** `last_update`, `conversation_start`

**`StateTracker.update()`** — Stateless analyzer that detects:
1. Emotional tone (positive/frustrated/curious/playful signals)
2. Repetition (Jaccard word-overlap ≥ 0.7 threshold)
3. Testing/adversarial behavior (jailbreak, ignore instructions, etc.)
4. Meta-commentary ("you already said", "that's not helpful", etc.)
5. Interaction pattern (rapid_fire/exploratory/deep_dive/standard)
6. Dynamic personality mode based on accumulated signals

**In-memory cache:** `_state_cache` with LRU eviction at 200 entries. DB persistence handled by `query_db.save_conversation_state()`.

**Public API:**
| Function | Purpose |
|---|---|
| `get_or_create_state(cid)` | Cache lookup or create fresh state |
| `set_state(cid, state)` | Store/update state in cache |
| `clear_state(cid)` | Remove from cache (on conversation delete) |

---

### behavior_engine.py — Behavioral Routing

**Purpose:** Sits between intent classification and retrieval — modulates *how* the pipeline responds based on conversational state. This is what makes the difference between a search engine and a conversational partner.

**`BehaviorDecision` dataclass** — output of the engine:
| Field | Description |
|---|---|
| `behavior_mode` | One of 8 modes: standard, greeting, repetition_aware, testing_aware, meta_aware, frustration_recovery, rapid_fire, exploratory |
| `skip_retrieval` | Skip RAG/QA entirely (greetings, meta-comments) |
| `reduce_retrieval` / `boost_retrieval` | Modulate retrieval depth |
| `rag_k_override` / `rag_min_similarity_override` | Override policy's retrieval params |
| `personality_mode` | Override: default, concise, detailed, playful, empathetic |
| `response_length_hint` | Override: brief, normal, detailed |
| `behavior_context` | Text injected into BEHAVIOR_STATE_FRAME |
| `meta_instruction` | Specific behavioral instruction for LLM |
| `triggers` | List of detected trigger labels |

**`BehaviorEngine.evaluate()`** — Priority-ordered detection:
1. **Frustration recovery** → empathetic personality, boost retrieval
2. **Testing/adversarial** → concise personality, skip retrieval
3. **Meta-commentary** → acknowledge, skip retrieval
4. **Repetition** → rephrase, reduce retrieval
5. **Greeting** → playful personality, skip retrieval
6. **Rapid-fire** → concise personality, reduce retrieval
7. **Exploratory** → detailed personality, boost retrieval
8. **Tone overlays** → personality adjustments based on emotional tone
9. **Standard** → no overrides

---

### llm/ Package

#### llm/client.py — Provider Wrapper

Thin wrapper that delegates to whichever provider is configured via `LLM_PROVIDER`:

- `completion(messages, temperature, max_tokens) → str`
- `stream_text_deltas(messages, temperature, max_tokens) → Generator[str]`

Re-exports token budgets: `MAX_RESPONSE_TOKENS`, `MAX_CLASSIFIER_TOKENS`, `MAX_PROFILE_DETECT_TOKENS`, `MAX_TITLE_TOKENS`.

#### llm/classifier.py — Intent Classification

**5-intent taxonomy:** `general`, `continuation`, `knowledge_base`, `profile`, `privacy`

**Pre-heuristic fast paths (no LLM round-trip):**

| Pattern | Intent | Confidence | Condition |
|---|---|---|---|
| Greeting | `general` | 0.97 | ≤8 words, matches GREETING_PATTERNS |
| Profile statement | `profile` | 0.92 | Starts with PROFILE_STATEMENT_PREFIXES, no `?`, ≤15 words |
| Privacy phrase | `privacy` | 0.95 | Contains any PRIVACY_SIGNALS substring |
| Short pronoun follow-up | `continuation` | 0.85 | ≤8 words, has continuation pronoun + question mark, or signal word and ≤4 words |

**LLM fallback:** When no heuristic matches, sends recent conversation context + INTENT_PROMPT to the LLM. Response is JSON-parsed with robust extraction (`{"intent": "...", "confidence": 0.X}`). Unknown intents fall back to `general`.

**Cache integration:** Results cached in Redis (when enabled) to avoid repeated classification of identical queries.

#### llm/generators.py — Response Generation

| Function | Purpose |
|---|---|
| `generate_response(...)` | Non-streaming — calls `build_messages()` then `completion()` |
| `generate_response_stream(...)` | Streaming — yields Vercel AI SDK data-stream lines (`0:"text"\n`) |
| `generate_title(user_message)` | 3-6 word title from first message, word-boundary truncation at 50 chars |

Both generation functions accept the same parameters: `user_query`, `chat_history`, `rag_context`, `profile_context`, `similar_qa_context`, `curated_history`, `privacy_mode`, `greeting_name`, `behavior_context`, `meta_instruction`, `personality_mode`, `response_length_hint`.

**Streaming protocol:**
```
0:"token"                              ← text delta
0:" next token"                        ← text delta
e:{"finishReason":"stop"}              ← finish event
d:{"finishReason":"stop"}              ← done signal
```

#### llm/prompts.py — Prompt Templates

**Single source of truth** for all text that becomes LLM system/user messages.

| Constant | Purpose |
|---|---|
| `INTENT_PROMPT` | Classification rules — 5 intents with decision rules and confidence guidelines |
| `SYSTEM_PROMPT` | Main system instructions — accuracy, context usage, tone, formatting, continuity, profile, length, safety |
| `PROFILE_CONTEXT_FRAME` | Frame for injecting user profile data |
| `RAG_CONTEXT_FRAME` | Frame for injecting knowledge base excerpts |
| `QA_CONTEXT_FRAME` | Frame for injecting prior Q&A |
| `PRIVACY_QA_FRAME` | Transparency rules when user asks about data privacy |
| `GREETING_PERSONALIZATION_FRAME` | Instructions for personalized greeting with user's name |
| `PROFILE_DETECT_PROMPT` | Instructions for extracting personal facts from messages |
| `TITLE_PROMPT` | Instructions for generating 3-6 word conversation titles |
| `BEHAVIOR_STATE_FRAME` | Behavioral intelligence context frame (tone, patterns, meta-instruction) |
| `PERSONALITY_FRAMES` | Dict of 5 personality modes (default, concise, detailed, playful, empathetic) |
| `RESPONSE_LENGTH_HINTS` | Dict of 3 response length hints (brief, normal, detailed) |

#### llm/prompt_orchestrator.py — Message Assembly

`build_messages(user_query, *, chat_history, curated_history, rag_context, profile_context, similar_qa_context, privacy_mode, greeting_name) → list[dict]`

**Message ordering:**
1. System prompt (always first)
2. Greeting personalization frame (if `greeting_name` set)
3. Behavior state frame (if `behavior_context` or `meta_instruction` present — includes personality + length hint)
4. Profile context frame (if `profile_context` provided)
5. RAG context frame (if `rag_context` provided)
6. Privacy frame OR Q&A context frame (mutually exclusive)
7. Conversation history (budget-enforced, optionally summarized)
8. Current user message (always last)

**Token budgeting:**
- Computes preamble tokens (all system messages + user query)
- Calls `compute_history_budget()` to determine remaining space
- Takes the minimum of `MAX_HISTORY_TOKENS` and the dynamic budget
- When `ENABLE_HISTORY_SUMMARIZATION=True`: uses `summarize_old_turns()`
- Otherwise: uses `fit_messages_to_budget()` for hard trimming

#### llm/profile_detector.py — Personal Info Extraction

`detect_profile_updates(user_message, assistant_response) → list[dict]`

**Two-gate approach:**
1. **Signal gate:** Checks for 33 personal signals (e.g., "my name", "i work", "i prefer"). If none found → returns `[]` immediately (no LLM call).
2. **LLM extraction:** Sends message + response to LLM with `PROFILE_DETECT_PROMPT`. Returns `[{"key": "snake_case", "value": "...", "category": "..."}]`.

Categories: `personal`, `professional`, `preferences`, `health`, `education`, `other`.

#### llm/providers/ — Pluggable Providers

**`base.py` — Abstract base class:**
```python
class LLMProvider(ABC):
    name: str                                       # Provider identifier
    complete(messages, temperature, max_tokens) → str
    stream_text_deltas(messages, temperature, max_tokens) → Generator[str]
```

**`__init__.py` — Dynamic loader:**
- Reads `LLM_PROVIDER` from settings
- Imports only the selected provider's SDK (lazy imports)
- Caches as singleton via `provider()` function
- `reset()` forces re-initialization

**Provider implementations:**

| Provider | SDK | Model Default | Special |
|---|---|---|---|
| `cerebras` | `cerebras-cloud-sdk` | `llama-4-scout-17b-16e-instruct` | Fastest inference |
| `openai` | `openai` | `gpt-4o-mini` | Also supports Azure, vLLM, Ollama via `LLM_BASE_URL` |
| `anthropic` | `anthropic` | `claude-sonnet-4-20250514` | Messages API |

---

## Pipeline Deep Dive

The pipeline is the core of the system. Every chat request (streaming or non-streaming) passes through `run_pipeline()`.

```
User message arrives
       │
       ▼
  ┌─ PARALLEL (ThreadPoolExecutor, 3 workers) ──┐
  │  1. Embed query (sentence-transformers)      │
  │  2. Load history (PostgreSQL)                │
  │  3. Load profile (PostgreSQL)                │
  └──────────────────────────────────────────────┘
       │
       ▼
  4. Classify intent
     ├── Cache hit? → return cached
     ├── Pre-heuristic match? → return immediately
     └── LLM fallback → classify + cache result
       │
       ▼
  5. Topic gate (continuation only)
     cosine_sim(query_embedding, topic_vector) < 0.35? → demote to "general"
       │
       ▼
  5b. Behavior engine
     ├── Load/create conversation state (cache + DB fallback)
     ├── Update state: tone, repetition, testing, meta, pattern
     └── Evaluate: 8 priority-ordered modes → BehaviorDecision
       │
       ▼
  6. Extract ContextFeatures + BehaviorPolicy.resolve() + behavior overrides + Hooks.policy_override()
       │
       ▼
  7. History pruning
     ├── Recency window: last 6 messages
     ├── Semantic retrieval: top 3 similar Q&A from same conversation
     └── Merge + deduplicate → curated_history
       │
       ▼
  8. Selective retrieval (driven by PolicyDecision flags)
     ├── inject_rag? → vector_store.search(query, k, min_similarity)
     ├── inject_qa_history? → search_similar_queries() + search_same_conversation_qa()
     ├── inject_profile? → format profile entries as text
     └── privacy_mode? → set privacy flag instead of Q&A
       │
       ▼
  9. Hooks.run_before_generation(pipeline_result)
       │
       ▼
  10. Generate response
      ├── /chat → complete(build_messages(...)) → JSON response
      └── /chat/stream → stream_text_deltas(build_messages(...)) → SSE
       │
       ▼
  11. Hooks.run_after_generation(response, pipeline_result)
      Hooks.run_before_persist(pipeline_result, response)
       │
       ▼
  12. Background persist (worker.submit)
      ├── save_chat_message (user + assistant)
      ├── save_user_query (with embedding)
      ├── detect_profile_updates → batch_update_profile
      ├── update_topic_vector (exponential moving average, α=0.3)
      └── generate_title (if first message in conversation)
```

---

## API Reference

### Chat

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Non-streaming chat — returns complete JSON |
| `POST` | `/chat/stream` | Streaming chat — Vercel AI SDK data stream over SSE |
| `POST` | `/chat/regenerate` | Delete last assistant message and regenerate |

**POST /chat** — Request:
```json
{
  "user_query": "What is RAG?",
  "conversation_id": "optional-uuid",
  "tags": ["optional"],
  "user_id": "default"
}
```
Response:
```json
{
  "response": "RAG stands for...",
  "conversation_id": "uuid",
  "intent": "knowledge_base",
  "confidence": 0.92,
  "retrieval_info": {"num_docs": 4, "similar_queries": 2, "behavior_mode": "standard", "behavior_triggers": ["standard"]},
  "query_tags": ["rag", "ai"],
  "behavior_mode": "standard"
}
```

**POST /chat/stream** — Same request body. Response is SSE:
```
8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]
8:[{"stage":"retrieved","retrieval_info":{"num_docs":4}}]
8:[{"stage":"generating"}]
0:"RAG "
0:"stands "
0:"for..."
8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...}}]
e:{"finishReason":"stop"}
d:{"finishReason":"stop"}
```

**POST /chat/regenerate** — Request:
```json
{
  "conversation_id": "uuid",
  "user_id": "default"
}
```

### Conversations

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/conversations` | List all conversations for user |
| `GET` | `/conversations/search?q=...&user_id=...` | Search conversations by title/content |
| `POST` | `/conversations` | Create new conversation |
| `GET` | `/conversations/{id}` | Get conversation messages |
| `GET` | `/conversations/{id}/export?format=json` | Export conversation (JSON or text) |
| `GET` | `/conversations/{id}/state` | Inspect behavioral state (debug) |
| `PUT` | `/conversations/{id}` | Rename conversation |
| `DELETE` | `/conversations/{id}` | Delete conversation |

### Profile

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/profile?user_id=...` | Get user profile entries |
| `POST` | `/profile` | Add profile entry |
| `PUT` | `/profile/{entry_id}` | Update profile entry |
| `DELETE` | `/profile/{entry_id}` | Delete profile entry |

### Utility

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check (DB status, doc count, provider info, version) |
| `GET` | `/` | Serve React frontend (or fallback HTML) |

---

## Database Schema

PostgreSQL 16 + pgvector extension. All tables created by `init_db()` on first startup.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    user_id TEXT NOT NULL DEFAULT 'default',
    tags TEXT[] DEFAULT '{}',
    topic_embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_queries (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    embedding vector(768),
    tags TEXT[] DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_profile (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, key)
);

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768),
    source TEXT DEFAULT 'default',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW indexes for fast vector search
CREATE INDEX ON user_queries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON conversations USING hnsw (topic_embedding vector_cosine_ops);

-- Behavioral intelligence state (per-conversation)
CREATE TABLE conversation_state (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    state_data      JSONB NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Configuration Reference

All settings are in `backend/settings.py`, driven by environment variables in `.env`:

```env
# ── LLM Provider ──────────────────────────────────────────
LLM_PROVIDER=cerebras                    # cerebras | openai | anthropic
LLM_API_KEY=your-api-key
LLM_MODEL=                               # empty = provider default
LLM_BASE_URL=                            # for vLLM, Azure, Ollama

# ── Token Budgets ─────────────────────────────────────────
MAX_CONTEXT_WINDOW=65536
MAX_RESPONSE_TOKENS=2048
MAX_HISTORY_TOKENS=8000
MAX_CLASSIFIER_TOKENS=60
MAX_PROFILE_DETECT_TOKENS=200
MAX_TITLE_TOKENS=30

# ── Embedding ─────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
QUERY_INSTRUCTION=Represent this sentence for searching relevant passages:

# ── Retrieval ─────────────────────────────────────────────
RETRIEVAL_K=4                            # docs per search
QA_K=4                                   # prior Q&A per search
SIMILARITY_THRESHOLD=0.3                 # minimum cosine for KB docs

# ── Continuity ────────────────────────────────────────────
TOPIC_CONTINUATION_THRESHOLD=0.35
RECENCY_WINDOW=6
SEMANTIC_K=3

# ── Knowledge Base ────────────────────────────────────────
KNOWLEDGE_DIR=knowledge                  # relative to project root
CHUNK_SIZE=500
CHUNK_OVERLAP=50
FORCE_REINDEX=false

# ── Database ──────────────────────────────────────────────
DATABASE_URL=postgresql://root:password@localhost:55432/chatapp
DB_POOL_MIN=2
DB_POOL_MAX=10

# ── Cache (optional) ─────────────────────────────────────
ENABLE_CACHE=false
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# ── Pipeline ──────────────────────────────────────────────
DEFAULT_USER_ID=default
HISTORY_FETCH_LIMIT=50
ENABLE_HISTORY_SUMMARIZATION=true

# ── Behavior Engine ───────────────────────────────────────
BEHAVIOR_ENGINE_ENABLED=true             # Enable behavioral intelligence layer
BEHAVIOR_REPETITION_THRESHOLD=0.7        # Jaccard word-overlap threshold
BEHAVIOR_PATTERN_WINDOW=10               # Messages to consider for pattern detection
BEHAVIOR_STATE_PERSIST=true              # Persist state to DB (vs in-memory only)

# ── Server ────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=*
```

---

## Test Suite

**190+ tests** across 8 test files in `backend/tests/`:

| File | Tests | Coverage |
|---|---|---|
| `test_chunker.py` | 27 | Chunk splitting: paragraphs, sentences, characters, overlap, edge cases, empty input |
| `test_classifier.py` | ~25 | Pre-heuristic paths (greeting, profile, privacy, continuation), LLM fallback, cache hit, unknown intents |
| `test_context_manager.py` | ~20 | Token estimation, budget fitting, progressive summarization, edge cases |
| `test_policy.py` | ~30 | All 5 intents, cross-intent overlays, greeting detection, personal ref signals, feature extraction |
| `test_prompt_orchestrator.py` | ~22 | Message ordering, frame injection, behavior frame position, personality modes, privacy suppression, history fallback |
| `test_settings.py` | ~12 | Defaults, env overrides, bool parsing, immutability, behavior engine settings |
| `test_conversation_state.py` | 30 | State tracking: tone detection, repetition, testing, meta, patterns, cache, serialization |
| `test_behavior_engine.py` | 24 | All 8 behavior modes, priority ordering, retrieval modulation, personality overrides |

**Running tests:**
```bash
cd <project root>
python -m pytest backend/tests -v
```

All tests use mocking (no real DB or LLM calls required). The `conftest.py` in `backend/tests/` adds `backend/` to `sys.path` so flat imports work.

---

## Deployment

### Docker Compose (recommended)

```bash
docker compose up --build
```

This starts:
- **postgres** — pgvector/pgvector:pg16 with health check
- **app** — The backend at `http://localhost:8000`

The Dockerfile:
1. Copies `backend/requirements.txt` and installs dependencies
2. Copies `backend/`, `knowledge/`, `frontend/dist/`
3. Sets `WORKDIR /app/backend` (so flat imports work)
4. Runs `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Health check: `curl -f http://localhost:8000/health`

### Local Development

```bash
# 1. Create venv and install deps
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt

# 2. Start PostgreSQL
docker compose up postgres -d

# 3. Init project
python backend/cli.py init
python backend/cli.py ingest

# 4. Run dev server
python backend/cli.py dev
# or
start_server.bat              # Windows
./start_server.sh             # Linux/macOS
```

### One-Click Scripts

- **Windows:** `start_server.bat` — runs `uvicorn main:app --app-dir backend`
- **Linux/macOS:** `start_server.sh` — same via bash
