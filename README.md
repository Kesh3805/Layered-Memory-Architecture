# RAG Chat App

A production-quality Retrieval-Augmented Generation (RAG) chat application with intent-gated selective retrieval, real-time token streaming, user profile memory, and a clean single-page UI.

## Features

- **Intent-gated pipeline** — classifies every query before deciding what to retrieve
  - `general` → LLM only, no retrieval
  - `continuation` → curated conversation history + semantic pruning
  - `knowledge_base` → FAISS document search + cross-conversation Q&A
  - `profile` → user profile injection (query) or update-only (statement)
  - `privacy` → transparent disclosure of stored data, offer deletion
- **Real-time token streaming** — tokens appear live as they're generated (Vercel AI SDK data stream protocol over SSE)
- **User profile memory** — background detection stores facts users explicitly share; injected transparently when relevant
- **Topic similarity gate** — rolling topic vector prevents false "continuation" matches across domain jumps
- **Semantic history pruning** — recency window + top-k semantic matches; never injects flat raw history
- **Cerebras LLM** — `gpt-oss-120b` (65 536-token context) via OpenAI-compatible Python SDK
- **FAISS + sentence-transformers** — local `all-MiniLM-L6-v2` embeddings, no additional API key needed
- **PostgreSQL + pgvector** — persistent conversation store, cross-conversation similarity search
- **Single-file UI** — plain HTML/CSS/JS, no build step

## Project Structure

```
.
├── main.py          # FastAPI app — intent pipeline, all HTTP endpoints
├── llm.py           # LLM calls — classifier, streaming + non-streaming response, profile detector, title generator
├── query_db.py      # PostgreSQL helpers — conversations, messages, profile, topic vector
├── vector_store.py  # FAISS wrapper — add and search document chunks
├── embeddings.py    # sentence-transformers embedding helper
├── index.html       # Single-page chat UI
├── data.txt         # Knowledge base source document (edit to add your own content)
├── requirements.txt
├── .env.example
├── start_server.bat # Windows quick-start
└── start_server.sh  # Linux/macOS quick-start
```

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd Chatapp
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
CEREBRAS_API_KEY=your_cerebras_api_key_here
DATABASE_URL=postgresql://root:password@localhost:55432/chatapp
# Optional: override model (default: gpt-oss-120b)
# CEREBRAS_MODEL=llama3.1-8b
```

Get a free Cerebras API key at [cloud.cerebras.ai](https://cloud.cerebras.ai).

### 3. Start PostgreSQL (Docker)

```bash
docker run -d \
  --name chatapp-postgres \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=chatapp \
  -p 55432:5432 \
  pgvector/pgvector:pg16
```

### 4. Index your knowledge base

Edit `data.txt` with your content, then run:

```bash
python embeddings.py
```

### 5. Start the server

**Option A — Docker Compose (PostgreSQL + app, one command):**

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). PostgreSQL starts automatically; no separate setup needed.

> **First build:** downloads the base image, installs dependencies, and caches the sentence-transformers model (~80 MB to `model_cache` volume). Subsequent `docker compose up` starts are fast.

**Option B — local dev (venv):**

```bash
# Windows
start_server.bat

# macOS/Linux
./start_server.sh

# Or directly:
uvicorn main:app --reload --port 8000
```

> When PostgreSQL isn't available the app falls back to in-memory mode automatically (conversations reset on restart).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Non-streaming chat — returns full JSON |
| `POST` | `/chat/stream` | **Streaming chat** — Vercel AI SDK SSE format |
| `GET` | `/conversations` | List all conversations |
| `POST` | `/conversations` | Create a new conversation |
| `GET` | `/conversations/{id}/messages` | Fetch message history |
| `GET` | `/profile` | Get stored user profile entries |
| `DELETE` | `/profile/{id}` | Delete a profile entry |

## Streaming Protocol

`/chat/stream` outputs the [Vercel AI SDK data stream protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol):

```
0:"token"\n                          — text delta
8:[{"intent":..., "retrieval_info":...}]\n  — metadata annotation
e:{"finishReason":"stop"}\n          — finish event
d:{"finishReason":"stop"}\n          — done
```

The frontend reads these chunks with `ReadableStream` and appends tokens live with a blinking cursor.

## Token Limits

| Model | Max context | Response cap |
|-------|-------------|--------------|
| `gpt-oss-120b` | 65 536 | 2 048 |
| `llama3.1-8b` | 8 192 | 2 048 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CEREBRAS_API_KEY` | — | **Required** |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `CEREBRAS_MODEL` | `gpt-oss-120b` | Model override |

## Troubleshooting

**Port already in use:**
```bash
uvicorn main:app --reload --port 8001
```

**FAISS not found:**
```bash
pip install faiss-cpu
```

**sentence-transformers import error:**
```bash
pip install sentence-transformers --upgrade
```

## License

MIT
