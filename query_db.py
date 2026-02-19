# query_db.py – PostgreSQL + pgvector: Intent-gated architecture
#
# Replaced 3-layer memory system with:
#   - user_profile (structured key/value identity data)
#   - user_queries  (per-query embeddings, used only for KB/continuation intents)
#   - conversations + chat_messages (unchanged)
#
# Removed: memories table, conversation summaries, memory extraction
#
import math
import os
import uuid
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import psycopg2
from psycopg2.extras import Json
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Connection config — DATABASE_URL takes priority (docker-compose / prod),
#  falls back to individual env vars for local dev.
# ---------------------------------------------------------------------------
_database_url = os.getenv("DATABASE_URL")
if _database_url:
    _p = urlparse(_database_url)
    DB_CONFIG = {
        "host": _p.hostname or "localhost",
        "port": _p.port or 5432,
        "database": (_p.path or "/chatapp").lstrip("/"),
        "user": _p.username or "root",
        "password": _p.password or "password",
    }
else:
    DB_CONFIG = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "55432")),
        "database": os.getenv("POSTGRES_DB", "chatapp"),
        "user": os.getenv("POSTGRES_USER", "root"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
    }

# Connection pool — replaces individual get_connection() calls
_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        _pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)
    return _pool


def get_connection():
    """Get a pooled connection. Caller must call put_connection() when done."""
    return _get_pool().getconn()


def put_connection(conn):
    """Return a connection to the pool."""
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
#  SCHEMA
# ═══════════════════════════════════════════════════════════════════

def init_db():
    """Create all tables needed for the intent-gated architecture."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # ── conversations ─────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL DEFAULT 'New Chat',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count   INTEGER DEFAULT 0,
                is_archived     BOOLEAN DEFAULT FALSE,
                metadata        JSONB DEFAULT '{}'
            );
        """)

        # Migration: add message_count if missing
        try:
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0;")
        except Exception:
            pass

        # Migration: add topic_embedding (rolling topic anchor vector)
        try:
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS topic_embedding vector(384);")
        except Exception:
            pass

        # ── user_profile (structured identity data) ───────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id          SERIAL PRIMARY KEY,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                category    TEXT DEFAULT 'general',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(key)
            );
        """)

        # ── user_queries (semantic search over past Q&A) ──────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id              SERIAL PRIMARY KEY,
                query_text      TEXT NOT NULL,
                embedding       vector(384),
                user_id         TEXT DEFAULT 'public',
                conversation_id TEXT,
                response_text   TEXT,
                tags            TEXT[] DEFAULT '{}',
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata        JSONB
            );
        """)

        # ── chat_messages ─────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id              SERIAL PRIMARY KEY,
                user_id         TEXT NOT NULL DEFAULT 'public',
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                tags            TEXT[] DEFAULT '{}',
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata        JSONB
            );
        """)

        # ── Indexes ───────────────────────────────────────────────
        idx = [
            "CREATE INDEX IF NOT EXISTS idx_queries_conv ON user_queries(conversation_id);",
            "CREATE INDEX IF NOT EXISTS idx_queries_ts   ON user_queries(timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_msgs_conv    ON chat_messages(conversation_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_convs_upd    ON conversations(updated_at DESC);",
        ]
        for ddl in idx:
            cur.execute(ddl)

        # Vector index for query embeddings (HNSW – works at any scale)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_emb
                ON user_queries USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception:
            pass

        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialized – intent-gated architecture ready")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  USER PROFILE  (structured identity data – replaces "memories")
# ═══════════════════════════════════════════════════════════════════

def get_user_profile() -> list:
    """Get all profile entries as a list of dicts."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, key, value, category, created_at, updated_at
            FROM user_profile ORDER BY category, key;
        """)
        rows = cur.fetchall()
        cur.close(); put_connection(conn)
        return [
            {"id": r[0], "key": r[1], "value": r[2], "category": r[3],
             "created_at": r[4].isoformat() if r[4] else None,
             "updated_at": r[5].isoformat() if r[5] else None}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return []


def get_profile_as_text() -> str:
    """Format the full user profile as a concise text block for LLM injection."""
    entries = get_user_profile()
    if not entries:
        return ""
    lines = []
    current_cat = None
    for e in entries:
        if e["category"] != current_cat:
            current_cat = e["category"]
            lines.append(f"[{current_cat}]")
        lines.append(f"  {e['key']}: {e['value']}")
    return "\n".join(lines)


def update_profile_entry(key: str, value: str, category: str = "general") -> int:
    """Upsert a profile entry (insert or update by key)."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_profile (key, value, category)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                category = EXCLUDED.category,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """, (key.strip(), value.strip(), category.strip()))
        pid = cur.fetchone()[0]
        conn.commit(); cur.close(); put_connection(conn)
        logger.info(f"Profile upserted #{pid}: {key} = {value[:60]}")
        return pid
    except Exception as e:
        logger.error(f"Error upserting profile entry: {e}")
        return None


def delete_profile_entry(entry_id: int) -> bool:
    """Delete a profile entry by ID."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM user_profile WHERE id = %s;", (entry_id,))
        ok = cur.rowcount > 0
        conn.commit(); cur.close(); put_connection(conn)
        return ok
    except Exception as e:
        logger.error(f"Error deleting profile entry: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  QUERY EMBEDDINGS  (semantic search – used selectively by intent)
# ═══════════════════════════════════════════════════════════════════

def _recency_score(ts):
    if not ts:
        return 0.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_h = max((now - ts).total_seconds() / 3600.0, 0.0)
    return math.exp(-age_h / 72.0)


def retrieve_similar_queries(query_embedding, k=5, conversation_id=None,
                             current_tags=None, min_similarity=0.25):
    """Retrieve semantically similar past Q&A across ALL conversations.
    
    min_similarity: threshold below which results are discarded.
    Prevents injecting irrelevant context when nothing good matches.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        cur.execute("""
            SELECT id, query_text, response_text, tags, timestamp,
                   1 - (embedding <=> %s::vector) AS similarity,
                   conversation_id
            FROM user_queries
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb, emb, max(k * 4, 16)))

        results = cur.fetchall()
        cur.close(); put_connection(conn)

        tag_set = set(current_tags or [])
        ranked = []
        for r in results:
            sim = float(r[5])
            if sim < min_similarity:
                continue  # Skip low-similarity noise
            tags = r[3] or []
            tag_overlap = len(tag_set & set(tags)) if tag_set else 0
            tag_sc = min(tag_overlap * 0.2, 0.4)
            rec = _recency_score(r[4])
            same_conv = 0.05 if (conversation_id and r[6] == conversation_id) else 0.0
            score = 0.70 * sim + 0.18 * rec + 0.05 * tag_sc + same_conv
            ranked.append({
                "id": r[0], "query": r[1], "response": r[2],
                "tags": tags, "timestamp": r[4],
                "similarity": sim, "recency": rec, "score": score,
                "conversation_id": r[6],
            })

        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:k]

    except Exception as e:
        logger.error(f"Error retrieving similar queries: {e}")
        return []


def retrieve_same_conversation_queries(query_embedding, conversation_id, k=4, min_similarity=0.2):
    """Retrieve similar past Q&A from the SAME conversation only."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        cur.execute("""
            SELECT id, query_text, response_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM user_queries
            WHERE conversation_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb, conversation_id, emb, k))

        results = cur.fetchall()
        cur.close(); put_connection(conn)

        return [
            {"id": r[0], "query": r[1], "response": r[2], "similarity": float(r[3])}
            for r in results if float(r[3]) >= min_similarity
        ]

    except Exception as e:
        logger.error(f"Error retrieving same-conv queries: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
#  TOPIC STATE  (rolling topic anchor – prevents cross-topic bleed)
# ═══════════════════════════════════════════════════════════════════

def get_topic_vector(conversation_id: str):
    """Return the current topic embedding for a conversation, or None."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT topic_embedding FROM conversations WHERE id = %s;",
            (conversation_id,)
        )
        row = cur.fetchone()
        cur.close(); put_connection(conn)
        if row and row[0] is not None:
            # psycopg2 returns pgvector as a list-like string; convert to np array
            vec = row[0]
            if isinstance(vec, str):
                vec = [float(x) for x in vec.strip("[]").split(",")]
            return np.array(vec, dtype=np.float32)
        return None
    except Exception as e:
        logger.error(f"Error getting topic vector: {e}")
        return None


def update_topic_vector(conversation_id: str, new_embedding, alpha: float = 0.1):
    """Update rolling topic vector: topic = (1-α)*old + α*new.
    
    On first message, sets the topic vector directly.
    alpha=0.1 means the topic shifts slowly – 10% weight to each new message.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Get current topic vector
        cur.execute(
            "SELECT topic_embedding FROM conversations WHERE id = %s;",
            (conversation_id,)
        )
        row = cur.fetchone()

        new_emb = new_embedding
        if isinstance(new_emb, np.ndarray):
            new_emb = new_emb.tolist()

        if row and row[0] is not None:
            old_vec = row[0]
            if isinstance(old_vec, str):
                old_vec = [float(x) for x in old_vec.strip("[]").split(",")]
            old_arr = np.array(old_vec, dtype=np.float32)
            new_arr = np.array(new_emb, dtype=np.float32)
            blended = ((1.0 - alpha) * old_arr + alpha * new_arr).tolist()
        else:
            blended = new_emb  # First message: set directly

        cur.execute(
            "UPDATE conversations SET topic_embedding = %s::vector WHERE id = %s;",
            (blended, conversation_id)
        )
        conn.commit(); cur.close(); put_connection(conn)
    except Exception as e:
        logger.error(f"Error updating topic vector: {e}")


def get_similar_messages_in_conversation(query_embedding, conversation_id: str,
                                          k: int = 3, min_similarity: float = 0.4):
    """Retrieve the top-k most semantically similar past Q&A turns within a
    single conversation. Used to supplement recency window with relevance.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        cur.execute("""
            SELECT query_text, response_text,
                   1 - (embedding <=> %s::vector) AS similarity,
                   timestamp
            FROM user_queries
            WHERE conversation_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (emb, conversation_id, emb, k * 3))

        results = cur.fetchall()
        cur.close(); put_connection(conn)

        return [
            {"query": r[0], "response": r[1], "similarity": float(r[2])}
            for r in results if float(r[2]) >= min_similarity
        ][:k]

    except Exception as e:
        logger.error(f"Error retrieving similar conv messages: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
#  TAG INFERENCE
# ═══════════════════════════════════════════════════════════════════

def infer_tags(query_text):
    text = (query_text or "").lower()
    tags = []
    kw = {
        "user_info": ["my name", "i am", "i'm", "my preference", "i prefer", "i like", "i work", "i live"],
        "topic_rag": ["rag"],
        "topic_faiss": ["faiss"],
        "topic_embeddings": ["embedding"],
        "topic_memory": ["memory", "remember"],
        "topic_db": ["postgres", "pgvector", "database"],
        "topic_code": ["code", "python", "javascript", "function", "class", "def ", "import "],
    }
    for tag, phrases in kw.items():
        if any(p in text for p in phrases):
            tags.append(tag)
    return tags or ["general"]


# ═══════════════════════════════════════════════════════════════════
#  CONVERSATION CRUD
# ═══════════════════════════════════════════════════════════════════

def create_conversation(title="New Chat"):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cid = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO conversations (id, title) VALUES (%s, %s) RETURNING id, title, created_at;",
            (cid, title),
        )
        r = cur.fetchone()
        conn.commit(); cur.close(); put_connection(conn)
        return {"id": r[0], "title": r[1], "created_at": r[2].isoformat() if r[2] else None}
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return None


def list_conversations(limit=50):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, created_at, updated_at, COALESCE(message_count, 0)
            FROM conversations
            WHERE is_archived = FALSE
            ORDER BY updated_at DESC
            LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
        cur.close(); put_connection(conn)
        return [
            {"id": r[0], "title": r[1],
             "created_at": r[2].isoformat() if r[2] else None,
             "updated_at": r[3].isoformat() if r[3] else None,
             "message_count": r[4]}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return []


def get_conversation(conversation_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = %s;", (conversation_id,))
        r = cur.fetchone()
        cur.close(); put_connection(conn)
        if not r:
            return None
        return {"id": r[0], "title": r[1],
                "created_at": r[2].isoformat() if r[2] else None,
                "updated_at": r[3].isoformat() if r[3] else None}
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return None


def rename_conversation(conversation_id, new_title):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s RETURNING id, title;",
            (new_title, conversation_id),
        )
        r = cur.fetchone()
        conn.commit(); cur.close(); put_connection(conn)
        return {"id": r[0], "title": r[1]} if r else None
    except Exception as e:
        logger.error(f"Error renaming conversation: {e}")
        return None


def delete_conversation(conversation_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_messages WHERE conversation_id = %s;", (conversation_id,))
        cur.execute("DELETE FROM user_queries WHERE conversation_id = %s;", (conversation_id,))
        cur.execute("DELETE FROM conversations WHERE id = %s;", (conversation_id,))
        ok = cur.rowcount > 0
        conn.commit(); cur.close(); put_connection(conn)
        return ok
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return False


def touch_conversation(conversation_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=%s;", (conversation_id,))
        conn.commit(); cur.close(); put_connection(conn)
    except Exception as e:
        logger.error(f"Error touching conversation: {e}")


def increment_message_count(conversation_id: str, amount: int = 1):
    """Increment the message counter on a conversation."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE conversations
            SET message_count = COALESCE(message_count, 0) + %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING message_count;
        """, (amount, conversation_id))
        row = cur.fetchone()
        conn.commit(); cur.close(); put_connection(conn)
        return row[0] if row else 0
    except Exception as e:
        logger.error(f"Error incrementing msg count: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════
#  MESSAGE STORAGE & RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

def store_query(query_text, embedding, response_text="", conversation_id=None,
                user_id="public", metadata=None, tags=None):
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        cur.execute("""
            INSERT INTO user_queries
            (query_text, embedding, user_id, conversation_id, response_text, tags, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """, (query_text, emb, user_id, conversation_id, response_text,
              tags or infer_tags(query_text), Json(metadata or {})))
        qid = cur.fetchone()[0]
        conn.commit(); cur.close(); put_connection(conn)
        return qid
    except Exception as e:
        logger.error(f"Error storing query: {e}")
        return None


def store_chat_message(role, content, conversation_id=None,
                       user_id="public", tags=None, metadata=None):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_messages (user_id, conversation_id, role, content, tags, metadata)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
        """, (user_id, conversation_id, role, content, tags or ["general"], Json(metadata or {})))
        mid = cur.fetchone()[0]
        conn.commit(); cur.close(); put_connection(conn)
        return mid
    except Exception as e:
        logger.error(f"Error storing chat message: {e}")
        return None


def get_conversation_messages(conversation_id, limit=200):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, tags, timestamp, id
            FROM chat_messages WHERE conversation_id = %s
            ORDER BY timestamp ASC LIMIT %s;
        """, (conversation_id, limit))
        rows = cur.fetchall()
        cur.close(); put_connection(conn)
        return [
            {"role": r[0], "content": r[1], "tags": r[2] or [],
             "timestamp": r[3].isoformat() if r[3] else None, "id": r[4]}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error getting conversation messages: {e}")
        return []


def get_recent_chat_messages(conversation_id, limit=10):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, tags, timestamp
            FROM chat_messages WHERE conversation_id = %s
            ORDER BY timestamp DESC LIMIT %s;
        """, (conversation_id, limit))
        rows = cur.fetchall()
        cur.close(); put_connection(conn)
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        logger.error(f"Error getting recent msgs: {e}")
        return []


def get_first_user_message(conversation_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT content FROM chat_messages
            WHERE conversation_id=%s AND role='user'
            ORDER BY timestamp ASC LIMIT 1;
        """, (conversation_id,))
        r = cur.fetchone()
        cur.close(); put_connection(conn)
        return r[0] if r else None
    except Exception as e:
        logger.error(f"Error getting first msg: {e}")
        return None
