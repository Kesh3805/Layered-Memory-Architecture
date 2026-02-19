# RAG Chat — Architecture & How It Works

A ChatGPT-style conversational AI with **policy-driven intent-gated selective retrieval**, built on FastAPI, Cerebras LLM, FAISS, PostgreSQL, and React.

---

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   React Frontend (frontend/)                     │
│  Sidebar · Chat UI · Profile Modal · Memory Panel · Streaming    │
│  Vite + Tailwind + Vercel AI SDK (useChat) + Zustand             │
└──────────────────────┬───────────────────────────────────────────┘
                       │  HTTP / SSE (Vercel AI SDK data stream)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI  (main.py)  v3.0.0                  │
│                                                                  │
│  POST /chat/stream ──► run_pipeline() ──► generate_response_     │
│  POST /chat         │                     stream / response()    │
│                     │                                            │
│  Conversations CRUD │  Profile CRUD   │  Serve UI (GET /)        │
└─────────┬───────────┴────────┬────────┴──────────────────────────┘
          │                    │
  ┌───────▼───────┐      ┌────▼─────┐       ┌───────────────┐
  │  llm/ package │      │query_db  │       │ vector_store   │
  │  ├ client     │      │ Postgres │       │   FAISS        │
  │  ├ classifier │      │ pgvector │       │  (in-memory)   │
  │  ├ orchestr.  │      └──────────┘       └───────────────┘
  │  ├ generators │
  │  └ profiler   │      ┌──────────┐
  └───────────────┘      │ policy.py│  ← BehaviorPolicy engine
                         └──────────┘
```

---

## File-by-File Breakdown

### `main.py` — Application Entry Point & Pipeline

**What it does:** Defines the FastAPI application, all HTTP endpoints, and the core
intent-gated pipeline that every chat message passes through.

**Key components:**

| Component | Purpose |
|---|---|
| `run_pipeline(request)` | The heart of the app. Takes a user message and returns a `PipelineResult` with intent classification, retrieved context, curated history, and embeddings — ready for the LLM. |
| `persist_after_response()` | Runs in a background thread after the response is sent. Stores messages, updates topic vectors, auto-generates titles, and detects profile updates. |
| `POST /chat/stream` | Streaming endpoint. Calls `run_pipeline()`, then streams tokens via Vercel AI SDK data stream protocol (SSE). |
| `POST /chat` | Non-streaming endpoint. Same pipeline, returns complete JSON. |
| Conversation CRUD | `POST/GET/PUT/DELETE /conversations` — create, list, rename, delete conversations. |
| Profile CRUD | `GET/POST/PUT/DELETE /profile` — manage user profile entries. |
| `GET /` | Serves the single-page `index.html` UI. |

**Pipeline flow (what happens on every message):**

```
User sends message
       │
       ▼
  1. EMBED — Convert query to a 384-dim vector (sentence-transformers)
       │
       ▼
  2. LOAD HISTORY — Fetch last 20 messages from this conversation (if DB available)
       │
       ▼
  3. CLASSIFY INTENT — LLM call + pre-heuristics determine the intent:
       │                 general | continuation | knowledge_base | profile | privacy
       │
       ▼
  4. TOPIC GATE — For "continuation" intent, compare query embedding to the
       │            conversation's rolling topic vector. If cosine similarity < 0.35,
       │            downgrade to "general" (prevents false continuation across topics)
       │
       ▼
  5. CONTEXT FEATURES + POLICY — Extract deterministic features from the query
       │   and current state (is_greeting, references_profile, has_profile_name, …).
       │   BehaviorPolicy resolves a PolicyDecision: what to inject, how to frame.
       │
       ▼
  6. PRUNE HISTORY — If policy says use_curated_history: keep last 6 messages
       │              (recency window). For long conversations, also retrieve top-3
       │              semantically similar older messages.
       │
       ▼
  7. RETRIEVAL (policy-driven) — Only fetch what the policy decision requests:
       │
       │   inject_rag=True      → FAISS vector search (k docs) + similar prior Q&A
       │   inject_qa_history    → Same-conversation Q&A for continuity
       │   inject_profile       → User profile data formatted as text
       │   privacy_mode         → Profile data + privacy transparency frame
       │   greeting_name="X"    → Greeting personalization frame with user's name
       │
       ▼
  8. GENERATE — Send assembled context to Cerebras LLM (stream or batch)
       │
       ▼
  9. PERSIST (background) — Store messages, update topic vector, auto-title,
                            detect & save profile updates
```

---

### `policy.py` — Behavior Policy Engine (NEW)

**What it does:** Deterministic rules that decide *what* context to inject and *how*
to frame the response.  Separates behavior logic from model-calling code.

**Key components:**

| Component | Purpose |
|---|---|
| `ContextFeatures` | Dataclass of computed features: is_greeting, references_profile, privacy_signal, is_followup, profile_name, etc. |
| `extract_context_features()` | Computes features from query text + intent + profile entries + conversation state. Pure, deterministic. |
| `PolicyDecision` | Dataclass of flags: inject_profile, inject_rag, inject_qa_history, privacy_mode, greeting_name, retrieval_route, etc. |
| `BehaviorPolicy.resolve()` | Takes features + intent, returns a PolicyDecision. Rules are evaluated in priority order (privacy > profile > knowledge_base > continuation > general) with cross-intent overlays (greeting personalization, personal-reference profile injection). |

**Why this exists:** Before, the old if/else chain in `run_pipeline()` mixed retrieval
decisions with prompt framing.  If a greeting didn't use the user's name, you'd need
to hunt through 200 lines of pipeline code.  Now: fix the rule in `BehaviorPolicy`.

---

### `llm/` package — LLM Interface (Cerebras)

**What it does:** All communication with the Cerebras API, split into single-responsibility modules.

| Module | Purpose |
|---|---|
| `client.py` | Pure API client: `Cerebras` init, `completion()` function, token budget constants. Zero behavior logic. |
| `classifier.py` | Intent classification: pre-heuristics (privacy signals, continuation pronouns) + LLM fallback. Returns `{"intent", "confidence"}`. |
| `prompts.py` | Single source of truth for ALL prompt templates: SYSTEM_PROMPT, INTENT_PROMPT, context frame templates (profile, RAG, Q&A, privacy, greeting), PROFILE_DETECT_PROMPT, TITLE_PROMPT. |
| `prompt_orchestrator.py` | `build_messages()` — assembles the OpenAI-format message list from a PolicyDecision. Injects greeting personalization, profile data, RAG context, privacy frame, Q&A, and conversation history in the correct order. |
| `generators.py` | `generate_response()`, `generate_response_stream()` (Vercel AI SDK data stream), `generate_title()`. All delegate to client + orchestrator. |
| `profile_detector.py` | `detect_profile_updates()` — pre-check with PERSONAL_SIGNALS list, then LLM extraction of structured {key, value, category} entries. |
| `__init__.py` | Re-exports for backward compatibility: `classify_intent`, `generate_response`, `generate_response_stream`, `generate_title`, `detect_profile_updates`. |

**System prompts (in order of injection by orchestrator):**

1. **SYSTEM_PROMPT** — Core personality and behavior rules (accuracy, formatting, profile/privacy handling)
2. **GREETING_PERSONALIZATION_FRAME** (conditional) — User's name for warm greeting
3. **PROFILE_CONTEXT_FRAME** (conditional) — User's stored profile data with strict usage rules
4. **RAG_CONTEXT_FRAME** (conditional) — Knowledge base documents retrieved from FAISS
5. **PRIVACY_QA_FRAME** or **QA_CONTEXT_FRAME** (conditional) — Privacy transparency rules or prior Q&A
6. **Conversation history** — Curated recent + semantically relevant older messages
7. **User query** — The actual message

---

### `query_db.py` — PostgreSQL + pgvector Database Layer

**What it does:** All database operations — connection pooling, table management, CRUD
for conversations/messages/profile/queries, and vector similarity search.

**Connection management:**
- Uses `psycopg2.pool.SimpleConnectionPool` (1-10 connections)
- Parses `DATABASE_URL` env var first, falls back to individual `POSTGRES_*` vars
- All functions use `get_connection()` / `put_connection()` for proper pooling

**Database tables:**

| Table | Purpose |
|---|---|
| `conversations` | Stores conversation metadata: title, timestamps, message count, rolling topic embedding (384-dim pgvector) |
| `chat_messages` | Individual messages with role (user/assistant), content, tags, metadata, timestamps |
| `user_queries` | Query embeddings + response text for semantic search across conversations |
| `user_profile` | Key-value store for user identity data (name, preferences, job, etc.) |

**Key operations:**

| Category | Functions |
|---|---|
| Conversations | `create_conversation`, `list_conversations`, `get_conversation`, `rename_conversation`, `delete_conversation`, `touch_conversation`, `increment_message_count` |
| Messages | `store_chat_message`, `get_conversation_messages`, `get_recent_chat_messages` |
| Queries | `store_query`, `retrieve_similar_queries`, `retrieve_same_conversation_queries` |
| Profile | `get_user_profile`, `get_profile_as_text`, `update_profile_entry`, `delete_profile_entry` |
| Topic | `get_topic_vector`, `update_topic_vector`, `get_similar_messages_in_conversation` |

---

### `embeddings.py` — Local Embedding Model

**What it does:** Loads the `all-MiniLM-L6-v2` sentence-transformer model and provides
a single function to embed text into 384-dimensional vectors.

```python
get_embedding(text: str) → np.ndarray  # shape: (384,), dtype: float32
```

This model runs **locally** (no API key needed). It's used for:
- Query embeddings (for the pipeline)
- Document embeddings (for FAISS indexing)
- Similarity comparisons (topic gate, semantic history retrieval)

---

### `vector_store.py` — FAISS In-Memory Vector Index

**What it does:** Maintains an in-memory FAISS `IndexFlatL2` index for the knowledge
base documents loaded from `data.txt`.

| Function | Purpose |
|---|---|
| `add_documents(chunks)` | Embeds text chunks and adds them to the FAISS index |
| `search(query, k=4)` | Returns the k most similar document chunks to the query |

The index is rebuilt on every server start from `data.txt`. It uses L2 (Euclidean)
distance for nearest-neighbor search.

---

### `data.txt` — Knowledge Base Source

A plain text file containing the private knowledge base content. At startup, `main.py`
reads this file, splits it into overlapping chunks (500 chars, 450-char stride), and
indexes them into FAISS.

**Current topics:** RAG Chat architecture, intent classification system, BehaviorPolicy
engine, LLM module structure, FAISS indexing, PostgreSQL pgvector schemas, streaming
protocol with stage events, React AI-native UI layer, debug mode, command palette,
profile management, sidebar intelligence, topic similarity tracking, and the full
pipeline flow from user query to response delivery.

To add your own knowledge: simply edit `data.txt` and restart the server. The content
will be automatically chunked, embedded, and indexed.

---

### `index.html` — Vanilla Fallback UI

**What it does:** Legacy chat interface served as a single HTML file (no build step).
Kept as a fallback — if `frontend/dist/` doesn't exist, FastAPI serves this instead.

---

### `frontend/` — React Frontend

**Stack:** React 18 + Vite + Tailwind CSS + Vercel AI SDK (`useChat`) + Zustand

**Architecture:**

```
frontend/
  src/
    main.tsx              — React entry point
    App.tsx               — Shell: sidebar + chat + profile modal + command palette
    index.css             — Tailwind directives + custom prose/code/animation styles
    api.ts                — Typed fetch wrappers for all backend endpoints
    store.ts              — Zustand store: conversations, profile, UI state,
                            debugMode, commandPaletteOpen
    hooks/
      use-chat-stream.ts  — Custom hook wrapping useChat with our backend format
    components/
      AIMessage.tsx       — AI-native message: streaming phases, status bar,
                            intent badge, retrieval panel, debug panel, token meter
      Sidebar.tsx         — Conversation list with category icons, debug toggle,
                            command palette shortcut, rename, delete, profile button
      ChatArea.tsx        — Messages container, scroll-to-bottom FAB, header bar
                            with debug mode toggle button
      InputArea.tsx       — Textarea with auto-resize, send/stop buttons
      WelcomeScreen.tsx   — Landing screen with suggestion cards
      ProfileModal.tsx    — View/add/delete profile entries
      CommandPalette.tsx  — Ctrl+K command palette: new chat, search, debug, profile
      MemoryPanel.tsx     — (Legacy) Expandable memory panel
      Message.tsx         — (Legacy) Basic message renderer, replaced by AIMessage
      ai/                 — AI-native component primitives:
        index.ts          — Barrel re-exports for all AI components
        AIIntentBadge.tsx — Color-coded intent badge with confidence dot
                            (green ≥0.85, yellow 0.6–0.84, gray <0.6)
        AIStatusBar.tsx   — Horizontal event timeline with clickable chips:
                            [Classified: intent] [Retrieved: N docs] [Similar Q&A]
                            [Topic: 0.61] [Profile injected] [Generating…/Complete]
        AIRetrievalPanel.tsx — Expandable drawer: FAISS sources, Q&A matches,
                               profile data, topic similarity, full retrieval info
        AITokenMeter.tsx  — Context window usage bar (estimated tokens / 65,536)
        AIDebugPanel.tsx  — Raw system internals: PolicyDecision JSON, retrieval_info,
                            query_tags, expandable JSON sections
```

**AI-Native UI Layer:**

The `components/ai/` folder contains primitives that surface backend intelligence:

| Component | What It Shows |
|---|---|
| `AIIntentBadge` | Intent label + confidence score with colored dot (green/yellow/gray) |
| `AIStatusBar` | Horizontal chip timeline of pipeline stages above each assistant message |
| `AIRetrievalPanel` | Full retrieval breakdown: document count, Q&A matches, profile injection, topic similarity |
| `AITokenMeter` | Visual bar showing estimated token usage relative to 65K context window |
| `AIDebugPanel` | Raw JSON of backend decisions — only visible when Debug Mode is on |

**Streaming Phases:**

Before text content arrives, the backend emits stage annotations via the data stream:
1. `8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]`
2. `8:[{"stage":"retrieved","retrieval_info":{...}}]` (conditional)
3. `8:[{"stage":"generating"}]`

The `AIMessage` component reads these from `message.annotations` and shows:
- "Thinking…" → "Retrieving context…" → "Preparing response…" → "Generating…"
- Each stage fills in a chip on the `AIStatusBar`

**Debug Mode:**

Toggled from sidebar footer or chat header. When active:
- Every assistant message gets an `AIDebugPanel` showing raw PolicyDecision JSON
- Expandable sections: Retrieval Info, Query Tags, Full Metadata
- Yellow indicator badge in sidebar and header

**Command Palette (Ctrl+K):**

- Centered modal with search input, fuzzy filtering
- Commands: New Chat, View Profile, Toggle Debug Mode + recent conversations
- Arrow key navigation + Enter to select + Esc to close

**Sidebar Intelligence:**

- Category icon detection from conversation titles via regex rules:
  Brain (ML/AI), Shield (Privacy), Database (SQL), Code2 (Programming),
  BookOpen (RAG/Embeddings), Globe (Web/Deploy), HelpCircle (General)
- Debug mode toggle with ON badge
- Command palette shortcut with Ctrl+K hint

**Key design decisions:**

| Decision | Rationale |
|---|---|
| `useChat` with custom `fetch` | AI SDK handles streaming state/parsing; we transform the request body from `{messages}` to `{user_query, conversation_id}` |
| `streamProtocol: 'data'` | Our backend already emits Vercel AI SDK data stream protocol (`0:`, `8:`, `e:`, `d:` lines) |
| Zustand over Redux | Minimal boilerplate, React 18 compatible, subscriptions are granular |
| Stage annotations (`8:[{stage}]`) | Pipeline stages stream in real-time, enabling the AI State Timeline without polling |
| `components/ai/` layer | AI primitives are separate from layout components, composable, reusable |
| Debug Mode as store flag | Single boolean propagates to every `AIMessage` without prop drilling |

**Development:** `cd frontend && npm install && npm run dev` — runs on port 5173

**Production:** `npm run build` writes to `frontend/dist/`; FastAPI serves it from `GET /`

---

### `docker-compose.yml` — Container Orchestration

```
services:
  postgres   — pgvector/pgvector:pg16 on port 55432
  app        — Python 3.12, FastAPI on port 8000
```

The `app` service depends on `postgres` being healthy before starting.
A HuggingFace model cache volume avoids re-downloading the sentence-transformer.

---

## Intent System — The Core Innovation

Unlike a basic chatbot that sends every message to the LLM with all available context,
this application **classifies intent first**, then **selectively retrieves only what's
needed**. This reduces noise, speeds up responses, and improves accuracy.

| Intent | What Gets Retrieved | LLM Receives |
|---|---|---|
| `general` | Nothing | System prompt + user query only |
| `continuation` | Same-conversation Q&A (semantic) | System prompt + curated history + similar Q&A + query |
| `knowledge_base` | FAISS docs + cross-conversation Q&A | System prompt + RAG docs + similar Q&A + history + query |
| `profile` (statement) | Nothing extra | System prompt + query (profile saved in background) |
| `profile` (question) | User profile data | System prompt + profile + query |
| `privacy` | User profile data + transparency instructions | System prompt + profile + privacy rules + query |

### Pre-heuristics (zero-latency shortcuts)

Before making the LLM classification call, the system checks:

1. **Privacy signals** — If the message contains phrases like "invasion of privacy",
   "do you store my data", etc., it immediately returns `privacy` intent without
   calling the LLM. (~40 signal phrases checked.)

2. **Short pronoun queries** — If the message is ≤5 words, contains conversation
   context, and includes words like "that", "it", "more", "elaborate" → immediately
   returns `continuation`. Saves an API call for quick follow-ups.

### Topic Continuation Gate

When a message is classified as `continuation`, the system checks if the query's
embedding is similar to the conversation's rolling topic vector. If cosine similarity
falls below **0.35**, the intent is downgraded to `general`. This prevents the LLM
from treating a new topic as a follow-up just because it came in the same conversation.

---

## Streaming Protocol

The app uses the **Vercel AI SDK data stream protocol** over Server-Sent Events:

```
8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]
                                    ← stage: classification complete
8:[{"stage":"retrieved","retrieval_info":{"num_docs":4,"similar_queries":2}}]
                                    ← stage: retrieval complete (conditional)
8:[{"stage":"generating"}]          ← stage: LLM generation starting
0:"FAISS"                           ← text delta (JSON-encoded string)
0:" is"                             ← another text delta
0:" a library"                      ← ...
8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...},"query_tags":[...]}]
                                    ← final metadata annotation
e:{"finishReason":"stop"}           ← finish event
d:{"finishReason":"stop"}           ← done event
```

**Stage events** are emitted before any text tokens, enabling the frontend to show
real-time pipeline progress: "Classifying…" → "Retrieving context…" → "Generating…"

The frontend reads **all** `8:` lines as `message.annotations[]`. Stage annotations
have a `stage` field; the final metadata annotation has an `intent` field. The
`AIMessage` component separates them for the status bar vs. debug panel.

---

## Profile System

The profile is a **key-value store** of stable personal facts:

```
name       → Keshav          (personal)
job_title  → ML Engineer     (work)
language   → Python          (preference)
framework  → FastAPI         (technical)
```

**How it gets populated:**
1. **Explicit statements** — When the user says "My name is Keshav", the `detect_profile_updates()` function
   extracts the fact after the response is generated (background, non-blocking).
2. **Manual entry** — The user can add/edit/delete entries via the Profile modal in the UI.

**How it gets used:**
- Profile data is injected as a system message ONLY when the pipeline detects a personal
  reference in the query ("What's my name?", "my job", etc.) or when the intent is `profile`.
- For `privacy` intent, profile data is always injected so the AI can transparently
  declare what is stored.

---

## Data Flow Diagram

```
User types "How does FAISS work?"
              │
              ▼
         index.html
    POST /chat/stream ─────────────────────────────────────────┐
              │                                                 │
              ▼                                                 │
         main.py                                                │
    run_pipeline() ──► embeddings.py                            │
         │                get_embedding("How does FAISS work?") │
         │                     │                                │
         │                     ▼                                │
         │              384-dim vector                          │
         │                     │                                │
         ▼                     │                                │
    classify_intent() ◄────────┘                                │
    (llm.py)                                                    │
         │ → intent: "knowledge_base"                           │
         │                                                      │
         ▼                                                      │
    vector_store.search()                                       │
    (FAISS: 4 nearest docs)                                     │
         │                                                      │
         ▼                                                      │
    query_db.retrieve_similar_queries()                         │
    (pgvector: similar past Q&A)                                │
         │                                                      │
         ▼                                                      │
    generate_response_stream()                                  │
    (llm.py → Cerebras API)                                     │
         │                                                      │
         ▼                                                      │
    SSE tokens stream ──────────────────────────────────────────┘
    0:"FAISS" 0:" is" 0:" a library"...        to the browser
         │
         ▼ (background thread)
    persist_after_response()
    → store_query() + store_chat_message() + update_topic_vector()
    → detect_profile_updates() + generate_title() (if first msg)
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `CEREBRAS_API_KEY` | **Yes** | — | API key for Cerebras LLM |
| `CEREBRAS_MODEL` | No | `gpt-oss-120b` | Model name (65,536 context window) |
| `DATABASE_URL` | No | — | Full PostgreSQL connection string |
| `POSTGRES_HOST` | No | `localhost` | DB host (only if DATABASE_URL not set) |
| `POSTGRES_PORT` | No | `55432` | DB port |
| `POSTGRES_DB` | No | `chatapp` | DB name |
| `POSTGRES_USER` | No | `root` | DB user |
| `POSTGRES_PASSWORD` | No | `password` | DB password |

---

## Running the Application

### Local (without Docker)

```bash
# 1. Start PostgreSQL with pgvector (e.g. via Docker)
docker run -d --name chatapp-postgres \
  -e POSTGRES_USER=root -e POSTGRES_PASSWORD=password -e POSTGRES_DB=chatapp \
  -p 55432:5432 pgvector/pgvector:pg16

# 2. Create .env with your API key
echo "CEREBRAS_API_KEY=your_key_here" > .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Open http://localhost:8000
```

### Docker Compose

```bash
# 1. Create .env with your API key
echo "CEREBRAS_API_KEY=your_key_here" > .env

# 2. Launch everything
docker compose up --build

# 3. Open http://localhost:8000
```

---

## Improvements Roadmap

### Performance

| Item | Impact | Effort |
|---|---|---|
| **Streaming latency** *(done)*: parallelize embed + DB loads with `ThreadPoolExecutor`; greetings and profile-statement pre-heuristics bypass the classifier LLM call | High | Low |
| **Async FastAPI** — convert `run_pipeline()` and the DB layer to `async def` + `asyncpg`. Currently every request blocks a thread from uvicorn's threadpool (FastAPI wraps sync routes automatically, but `asyncpg` would drastically cut DB wait time) | High | Medium |
| **Classifier caching** — LRU cache the top-N most common classification queries (greetings, "what is RAG", etc.) so repeated questions skip the LLM call entirely | Medium | Low |
| **FAISS persistence** — save the index to disk at shutdown and reload on startup. Currently `data.txt` is re-indexed from scratch on every restart (~0.5s startup cost). As the KB grows this will matter | Medium | Low |
| **Chunking by sentence** — replace the current character-stride chunking with sentence-boundary splits (e.g. `nltk.sent_tokenize`). Preserves semantic units, improves retrieval quality | Medium | Low |

### Retrieval Quality

| Item | Impact | Effort |
|---|---|---|
| **Hybrid search** — combine FAISS vector search with BM25 keyword search (sparse + dense), then re-rank with a cross-encoder. BM25 catches exact-term matches that embedding search misses | High | Medium |
| **Metadata filtering on FAISS** — tag documents by topic/source at indexing time; filter retrieval to the relevant subset. Reduces noise from unrelated chunks | High | Medium |
| **Multi-file knowledge base** — watch a `knowledge/` directory; index all `.txt` / `.md` / `.pdf` files automatically. Hot-reload on file change without restarting | Medium | Medium |
| **Re-ranking pass** — after FAISS top-k, run a lightweight cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`) to re-order by relevance before injecting into the prompt | High | Medium |
| **Better topic vector** — use an exponential moving average over the last N embeddings instead of a fixed-alpha update; decays older context faster | Low | Low |

### Memory & Profile

| Item | Impact | Effort |
|---|---|---|
| **Profile versioning** — store history of profile changes (key, old value, new value, timestamp) so users can see how their data evolved | Low | Low |
| **Profile confidence scores** — rate each profile entry by how explicitly it was stated ("My name is Alex" → high; "I usually work with Python" → medium) | Medium | Medium |
| **Conversation summarization** — for long conversations, run a summarization pass on messages older than N turns and store the summary embedding. Feed summaries instead of raw text for the semantic history retrieval | High | Medium |
| **Cross-session memory** — a "preferences" layer that captures stated preferences (e.g. "always use code examples") and injects them as a separate frame | Medium | Medium |

### UI / UX

| Item | Impact | Effort |
|---|---|---|
| **Message regeneration** — add a "Regenerate" button that re-sends the last user message with a higher temperature, storing both versions | Medium | Low |
| **Inline citation links** — when the response cites knowledge-base content, render `[1]` superscripts linking to the source chunk in the retrieval panel | High | Medium |
| **Conversation search** — full-text + semantic search across all conversations, surfaced in the Command Palette | High | Medium |
| **Response streaming cancel + resume** — track cancelled partial responses; offer "continue from here" | Low | High |
| **Keyboard shortcuts** — `N` new chat, `/` search, `D` debug mode, arrows for conversation navigation | Low | Low |
| **Export** — export full conversation as Markdown / PDF from the command palette | Low | Low |

### Reliability & Ops

| Item | Impact | Effort |
|---|---|---|
| **Rate limiting** — per-IP limits on `/chat/stream` using `slowapi` or a Redis token bucket | High | Low |
| **Structured logging** — replace `logger.info(f"…")` with `structlog` or JSON logging so logs are parseable in production | Medium | Low |
| **Health check endpoint** — `GET /health` returning DB status, FAISS index size, model load state | Low | Low |
| **Alembic migrations** — replace ad-hoc `init_db()` DDL with versioned Alembic migrations; safe to run on an existing production DB | High | Medium |
| **Unit tests** — pytest suite: classifier pre-heuristics, policy overlay correctness, pipeline result shapes, API contract tests | High | Medium |
| **E2E tests** — Playwright tests against the React frontend for the streaming chat flow | Medium | High |

---



| Layer | Technology |
|---|---|
| LLM | Cerebras `gpt-oss-120b` (65,536 context) |
| Embeddings | `all-MiniLM-L6-v2` (384-dim, local, no API key) |
| Vector search | FAISS `IndexFlatL2` (in-memory) |
| Database | PostgreSQL 16 + pgvector |
| Backend | FastAPI + Uvicorn (Python 3.12) |
| Frontend | React 18 + Vite + Tailwind CSS + Vercel AI SDK + Zustand |
| AI UI Layer | AIMessage, AIStatusBar, AIIntentBadge, AIRetrievalPanel, AIDebugPanel, AITokenMeter |
| Streaming | Vercel AI SDK data stream protocol over SSE (with stage events) |
| Containerization | Docker + Docker Compose |
