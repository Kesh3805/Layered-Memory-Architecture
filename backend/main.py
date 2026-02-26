"""FastAPI application — behavior-aware policy-driven intent-gated selective retrieval.

Architecture layers:
  1. Settings       (settings.py)  — centralized configuration
  2. Policy Engine  (policy.py)    — deterministic behavior rules
  3. Hooks          (hooks.py)     — extension points
  4. LLM Package    (llm/)         — pluggable providers, classifier, orchestrator
  5. Pipeline       (this file)    — orchestrates retrieval + generation
  6. Database       (query_db.py)  — persistence + vector search (pgvector)
  7. Vector Store   (vector_store.py) — pgvector-backed document index
  8. State Tracker  (conversation_state.py) — per-conversation behavioral state
  9. Behavior Engine(behavior_engine.py)    — behavioral routing layer

Pipeline (shared by /chat and /chat/stream):
  1. Embed query
  2. Load state     (history + profile entries)
  3. Classify intent
  4. Topic gate     (prevents false continuation across domain jumps)
  4b. Behavior engine (state tracking + behavioral routing)
  5. Extract features + policy resolve + behavior overrides + hooks
  6. History pruning
  7. Selective retrieval  (driven by policy decision, modulated by behavior)
  8. Generate response    (with behavior-aware prompt framing)
  9. Persist         (DB writes + state persistence via worker)
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import query_db
import vector_store
import worker
from chunker import chunk_text
from embeddings import get_query_embedding
from hooks import Hooks
from llm.classifier import classify_intent
from llm.generators import generate_response, generate_response_stream, generate_title
from llm.profile_detector import detect_profile_updates
from policy import BehaviorPolicy, extract_context_features
from settings import settings
from conversation_state import (
    ConversationState, StateTracker, get_or_create_state, set_state, clear_state,
)
from behavior_engine import BehaviorEngine, BehaviorDecision
from topic_threading import resolve_thread, get_thread_context
from research_memory import get_research_context, extract_concepts, link_concepts, extract_insights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Application lifespan
# ---------------------------------------------------------------------------
DB_ENABLED = False


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Run startup logic; yield to serve requests; clean up on shutdown."""
    global DB_ENABLED
    DB_ENABLED = query_db.init_db()
    vector_store.init(DB_ENABLED)
    if DB_ENABLED:
        logger.info("PostgreSQL connected — full persistence active")
    else:
        logger.warning("PostgreSQL not available — in-memory only")

    if not vector_store.has_documents() or settings.FORCE_REINDEX:
        _ingest_knowledge()
    else:
        logger.info(f"Document store ready ({vector_store.count()} chunks)")

    yield  # ← application runs here

    # Shutdown: drain background tasks
    worker.shutdown(wait=True)


# ---------------------------------------------------------------------------
#  App
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Chat", version="6.0.0", lifespan=lifespan)
_raw_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
_allowed_origins = _raw_origins if _raw_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_query: str
    conversation_id: Optional[str] = None
    tags: Optional[List[str]] = None
    user_id: str = settings.DEFAULT_USER_ID


class RenameRequest(BaseModel):
    title: str


class NewConversationRequest(BaseModel):
    title: str = "New Chat"


class ProfileEntryRequest(BaseModel):
    key: str
    value: str
    category: str = "general"
    user_id: str = settings.DEFAULT_USER_ID


class RegenerateRequest(BaseModel):
    conversation_id: str
    user_id: str = settings.DEFAULT_USER_ID


def _ingest_knowledge():
    """Read and index all files from the knowledge directory."""
    kb_dir = Path(settings.KNOWLEDGE_DIR)

    if kb_dir.exists():
        files = sorted(p for p in kb_dir.iterdir() if p.suffix in (".txt", ".md") and p.is_file())
        if files:
            if settings.FORCE_REINDEX:
                vector_store.clear()
            for path in files:
                text = path.read_text(encoding="utf-8")
                chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                vector_store.add_documents(chunks, source=path.name)
            logger.info(f"Indexed {len(files)} file(s) from {kb_dir}/")
            return

    logger.warning("No knowledge base found — add .txt or .md files to knowledge/ and run: python cli.py ingest")


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

# Shortcuts for frequently-accessed settings (avoids long dotted names).
_TOPIC_THRESHOLD = settings.TOPIC_CONTINUATION_THRESHOLD
_RECENCY_WINDOW = settings.RECENCY_WINDOW
_SEM_K = settings.SEMANTIC_K
_SIM_THRESHOLD = settings.SIMILARITY_THRESHOLD


@dataclass
class PipelineResult:
    """Output of the shared pipeline — everything needed for generation."""
    query: str
    cid: str
    query_embedding: Any
    intent: str
    confidence: float
    user_id: str = settings.DEFAULT_USER_ID
    rag_context: str = ""
    profile_context: str = ""
    similar_qa_context: str = ""
    curated_history: Optional[list] = None
    recent_messages: list = field(default_factory=list)
    retrieval_info: dict = field(default_factory=dict)
    query_tags: list = field(default_factory=list)
    privacy_mode: bool = False
    greeting_name: Optional[str] = None
    # ── Behavior engine outputs ───────────────────────────────────
    behavior_mode: str = "standard"
    behavior_context: str = ""
    meta_instruction: str = ""
    personality_mode: str = "default"
    precision_mode: str = "analytical"
    response_length_hint: str = "normal"
    # ── Research engine outputs ─────────────────────────────
    active_thread_id: str = ""
    thread_context: Optional[dict] = None
    research_context: Optional[dict] = None


def _profile_entries_to_text(entries: list[dict]) -> str:
    """Format profile entries as ``key: value`` lines for LLM injection."""
    if not entries:
        return ""
    return "\n".join(f"{e['key']}: {e['value']}" for e in entries)


def run_pipeline(request: ChatRequest) -> PipelineResult:
    """Execute the policy-driven intent-gated retrieval pipeline.

    Returns a PipelineResult with all context assembled — ready for
    either generate_response() or generate_response_stream().
    """
    cid = request.conversation_id or str(uuid.uuid4())
    query = request.user_query
    query_tags = request.tags or query_db.infer_tags(query)

    # Steps 1+2 (parallel): embed query AND load DB state simultaneously.
    # These three are fully independent — running them concurrently
    # cuts the pipeline preamble from ~90ms serial → ~50ms parallel.
    recent_messages: list = []
    profile_entries: list = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_embed = pool.submit(get_query_embedding, query)
        if DB_ENABLED:
            fut_history = pool.submit(query_db.get_recent_chat_messages, cid, settings.HISTORY_FETCH_LIMIT)
            fut_profile = pool.submit(query_db.get_user_profile, request.user_id)
    query_embedding = fut_embed.result()
    if DB_ENABLED:
        recent_messages = fut_history.result()
        profile_entries = fut_profile.result()

    # Step 3: Classify intent
    intent_result = classify_intent(query, recent_messages)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    logger.info(f"Intent: {intent} ({confidence:.2f})")

    # Step 4: Topic similarity gate (only for continuation)
    topic_similarity = None
    if DB_ENABLED and intent == "continuation":
        topic_vec = query_db.get_topic_vector(cid)
        if topic_vec is not None:
            q_arr = np.asarray(query_embedding, dtype=np.float32)
            dot = float(np.dot(q_arr, topic_vec))
            norm = float(np.linalg.norm(q_arr) * np.linalg.norm(topic_vec))
            topic_similarity = dot / norm if norm > 0 else 0.0
            if topic_similarity < _TOPIC_THRESHOLD:
                logger.info(f"Topic gate: {topic_similarity:.3f} < threshold → general")
                intent, confidence = "general", 0.7

    # Step 4b: Behavior engine — conversational state + behavioral routing
    behavior_decision = BehaviorDecision()  # default: standard mode
    if settings.BEHAVIOR_ENGINE_ENABLED:
        # Load / create conversation state
        conv_state = get_or_create_state(cid)
        # Try to load from DB if state is fresh (message_count == 0) and DB has data
        if DB_ENABLED and conv_state.message_count == 0:
            saved = query_db.get_conversation_state(cid)
            if saved:
                conv_state = ConversationState.from_dict(saved)
                set_state(cid, conv_state)

        # Extract recent user queries for repetition detection
        recent_user_queries = [
            m["content"] for m in recent_messages if m.get("role") == "user"
        ][-5:]

        # Update state with current message
        StateTracker.update(
            state=conv_state,
            query=query,
            intent=intent,
            confidence=confidence,
            recent_queries=recent_user_queries,
        )
        set_state(cid, conv_state)

        # Run behavior engine
        behavior_decision = BehaviorEngine.evaluate(conv_state, query, intent, confidence)
        logger.info(
            f"Behavior: mode={behavior_decision.behavior_mode}, "
            f"personality={behavior_decision.personality_mode}, "
            f"precision={behavior_decision.precision_mode}, "
            f"triggers={behavior_decision.triggers}, "
            f"skip_retrieval={behavior_decision.skip_retrieval}"
        )

    # Step 4c: Topic threading — resolve which thread this message belongs to
    active_thread_id = ""
    thread_context = None
    research_context_data = None
    if settings.THREAD_ENABLED and DB_ENABLED:
        thread_result = resolve_thread(cid, query_embedding, db_enabled=DB_ENABLED)
        active_thread_id = thread_result.thread_id
        if active_thread_id:
            thread_context = get_thread_context(cid, active_thread_id)
            logger.info(
                f"Thread: {active_thread_id[:8]}… "
                f"(new={thread_result.is_new}, sim={thread_result.similarity:.3f}, "
                f"msgs={thread_result.message_count})"
            )

    # Step 4d: Research context — gather related insights + concepts
    if settings.RESEARCH_INSIGHTS_ENABLED and DB_ENABLED:
        research_context_data = get_research_context(cid, query_embedding, active_thread_id, db_enabled=DB_ENABLED)

    # Step 5: Context features + policy resolve
    features = extract_context_features(
        query=query,
        intent=intent,
        profile_entries=profile_entries,
        conversation_length=len(recent_messages),
        topic_similarity=topic_similarity,
    )
    decision = BehaviorPolicy().resolve(features, intent)
    decision = Hooks.run_policy_override(features, decision)

    # Step 5b: Apply behavior decision to policy overrides
    if settings.BEHAVIOR_ENGINE_ENABLED:
        if behavior_decision.skip_retrieval:
            decision.inject_rag = False
            decision.inject_qa_history = False
            decision.retrieval_route = f"behavior:{behavior_decision.behavior_mode}"
        elif behavior_decision.reduce_retrieval:
            if behavior_decision.rag_k_override is not None:
                decision.rag_k = behavior_decision.rag_k_override
            if behavior_decision.rag_min_similarity_override is not None:
                decision.rag_min_similarity = behavior_decision.rag_min_similarity_override
        elif behavior_decision.boost_retrieval:
            if behavior_decision.rag_k_override is not None:
                decision.rag_k = behavior_decision.rag_k_override
            if behavior_decision.rag_min_similarity_override is not None:
                decision.rag_min_similarity = behavior_decision.rag_min_similarity_override

    logger.info(
        f"Policy: route={decision.retrieval_route}, profile={decision.inject_profile}, "
        f"rag={decision.inject_rag}, qa={decision.inject_qa_history}, "
        f"greeting={decision.greeting_name or 'no'}"
    )

    # Step 6: History pruning (conditional on policy)
    curated_history = None
    if decision.use_curated_history and recent_messages:
        recency_slice = recent_messages[-_RECENCY_WINDOW:]
        semantic_extra: list = []
        if DB_ENABLED and intent == "continuation" and len(recent_messages) > _RECENCY_WINDOW:
            older_qa = query_db.get_similar_messages_in_conversation(
                query_embedding, cid, k=_SEM_K, min_similarity=_SIM_THRESHOLD,
            )
            for item in older_qa:
                if item["similarity"] >= _SIM_THRESHOLD:
                    semantic_extra.append({"role": "user", "content": item["query"]})
                    if item.get("response"):
                        semantic_extra.append({"role": "assistant", "content": item["response"][:400]})
        curated_history = semantic_extra + recency_slice
        logger.info(
            f"History: {len(recent_messages)} raw → {len(curated_history)} curated "
            f"(recency={len(recency_slice)}, semantic={len(semantic_extra)})"
        )

    # Step 7: Selective retrieval (driven by policy decision)
    rag_context = ""
    profile_context = ""
    similar_qa_context = ""
    retrieval_info: dict = {
        "intent": intent,
        "confidence": confidence,
        "topic_similarity": round(topic_similarity, 3) if topic_similarity is not None else None,
        "route": decision.retrieval_route,
    }

    if decision.inject_rag:
        docs = vector_store.search(query, k=decision.rag_k, min_similarity=decision.rag_min_similarity)
        rag_context = "\n".join(docs)
        retrieval_info["num_docs"] = len(docs)
        if DB_ENABLED:
            similar_queries = query_db.retrieve_similar_queries(
                query_embedding, k=decision.qa_k, conversation_id=cid,
                current_tags=query_tags, min_similarity=decision.qa_min_similarity,
            )
            if similar_queries:
                lines = [
                    f"{i}. Q: {item['query']} → A: {(item.get('response') or '')[:200]}"
                    for i, item in enumerate(similar_queries[:4], 1)
                ]
                similar_qa_context = "\n".join(lines)
                retrieval_info["similar_queries"] = len(similar_queries)

    if decision.inject_qa_history and DB_ENABLED and len(recent_messages) > 2:
        same_conv_qa = query_db.retrieve_same_conversation_queries(
            query_embedding, cid, k=3, min_similarity=_SIM_THRESHOLD,
        )
        if same_conv_qa:
            lines = [f"- Q: {q['query'][:150]} → A: {(q['response'] or '')[:150]}" for q in same_conv_qa]
            similar_qa_context = "\n".join(lines)
            retrieval_info["same_conv_qa"] = len(same_conv_qa)

    if decision.inject_profile:
        profile_context = _profile_entries_to_text(profile_entries)
        retrieval_info["profile_injected"] = bool(profile_context)

    if decision.privacy_mode and not profile_context and DB_ENABLED:
        profile_context = _profile_entries_to_text(profile_entries)
        retrieval_info["profile_injected"] = bool(profile_context)

    if decision.greeting_name:
        retrieval_info["greeting_personalized"] = True

    # Inject behavior engine metadata into retrieval_info for observability
    if settings.BEHAVIOR_ENGINE_ENABLED:
        retrieval_info["behavior_mode"] = behavior_decision.behavior_mode
        retrieval_info["behavior_triggers"] = behavior_decision.triggers
        retrieval_info["personality_mode"] = behavior_decision.personality_mode
        retrieval_info["precision_mode"] = behavior_decision.precision_mode
        retrieval_info["response_length_hint"] = behavior_decision.response_length_hint

    # Inject research engine metadata
    if active_thread_id:
        retrieval_info["active_thread_id"] = active_thread_id
    if research_context_data:
        retrieval_info["research_insights_count"] = len(research_context_data.get("related_insights", []))
        retrieval_info["concept_links_count"] = len(research_context_data.get("concept_links", []))

    return Hooks.run_before_generation(PipelineResult(
        query=query,
        cid=cid,
        query_embedding=query_embedding,
        intent=intent,
        confidence=confidence,
        user_id=request.user_id,
        rag_context=rag_context,
        profile_context=profile_context,
        similar_qa_context=similar_qa_context,
        curated_history=curated_history,
        recent_messages=recent_messages,
        retrieval_info=retrieval_info,
        query_tags=query_tags,
        privacy_mode=decision.privacy_mode,
        greeting_name=decision.greeting_name,
        behavior_mode=behavior_decision.behavior_mode,
        behavior_context=behavior_decision.behavior_context,
        meta_instruction=behavior_decision.meta_instruction,
        personality_mode=behavior_decision.personality_mode,
        precision_mode=behavior_decision.precision_mode,
        response_length_hint=behavior_decision.response_length_hint,
        active_thread_id=active_thread_id,
        thread_context=thread_context,
        research_context=research_context_data,
    ))


def persist_after_response(p: PipelineResult, response_text: str):
    """Write messages / profile updates to DB via background worker."""
    if not DB_ENABLED:
        return

    response_text = Hooks.run_after_generation(response_text, p)
    Hooks.run_before_persist(p, response_text)

    def _work():
        try:
            existing = query_db.get_conversation_messages(p.cid, limit=1)
            is_first = len(existing) == 0

            query_db.store_query(
                query_text=p.query,
                embedding=p.query_embedding,
                response_text=response_text,
                conversation_id=p.cid,
                user_id=p.user_id,
                tags=p.query_tags,
                metadata={"intent": p.intent, "tags": p.query_tags},
            )
            query_db.update_topic_vector(p.cid, p.query_embedding, alpha=settings.TOPIC_DECAY_ALPHA)
            query_db.store_chat_message(
                role="user", content=p.query, conversation_id=p.cid,
                user_id=p.user_id,
                tags=p.query_tags, metadata={"intent": p.intent},
            )
            query_db.store_chat_message(
                role="assistant", content=response_text, conversation_id=p.cid,
                user_id=p.user_id,
                metadata={"intent": p.intent},
            )
            query_db.increment_message_count(p.cid, 2)
            query_db.touch_conversation(p.cid)

            if is_first:
                try:
                    title = generate_title(p.query)
                    query_db.rename_conversation(p.cid, title)
                    logger.info(f"Auto-titled: {title}")
                except Exception as e:
                    logger.error(f"Title gen failed: {e}")

            updates = detect_profile_updates(p.query, response_text)
            for entry in updates:
                query_db.update_profile_entry(
                    entry["key"], entry["value"],
                    category=entry.get("category", "general"),
                    user_id=p.user_id,
                )

            # Persist conversation state for behavioral intelligence
            if settings.BEHAVIOR_ENGINE_ENABLED and settings.BEHAVIOR_STATE_PERSIST:
                from conversation_state import get_or_create_state as _get_state
                conv_st = _get_state(p.cid)
                query_db.save_conversation_state(p.cid, conv_st.to_dict())

            # Research extraction: insights + concepts (async-safe, non-blocking)
            if settings.RESEARCH_INSIGHTS_ENABLED:
                try:
                    from llm.client import completion as _completion
                    insights = extract_insights(
                        p.query, response_text,
                        conversation_id=p.cid,
                        thread_id=p.active_thread_id or None,
                    )
                except Exception as e:
                    logger.error(f"Insight extraction error: {e}")

            if settings.CONCEPT_LINKING_ENABLED:
                try:
                    combined = f"{p.query} {response_text}"
                    concepts = extract_concepts(combined)
                    if concepts:
                        link_concepts(
                            concepts,
                            source_type="message",
                            source_id=p.cid,
                            conversation_id=p.cid,
                            thread_id=p.active_thread_id or None,
                        )
                except Exception as e:
                    logger.error(f"Concept linking error: {e}")

            # Thread summarization check
            if settings.THREAD_ENABLED and p.active_thread_id:
                try:
                    from thread_summarizer import maybe_summarize
                    maybe_summarize(p.active_thread_id, p.cid)
                except Exception as e:
                    logger.error(f"Thread summary error: {e}")
        except Exception as e:
            logger.error(f"Persist error: {e}")

    worker.submit(_work)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVERSATION CRUD
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/conversations")
def create_conversation(req: NewConversationRequest):
    if not DB_ENABLED:
        stub_id = str(uuid.uuid4())
        return {"id": stub_id, "title": req.title, "created_at": None, "updated_at": None, "message_count": 0}
    conv = query_db.create_conversation(title=req.title)
    if not conv:
        raise HTTPException(500, "Failed to create conversation")
    return conv


@app.get("/conversations")
def list_conversations(limit: int = 50):
    if not DB_ENABLED:
        return {"conversations": [], "count": 0}
    convs = query_db.list_conversations(limit=limit)
    return {"conversations": convs, "count": len(convs)}


@app.get("/conversations/search")
def search_conversations(q: str, limit: int = 20):
    """Full-text search across all conversation messages."""
    if not DB_ENABLED:
        return {"conversations": [], "count": 0}
    convs = query_db.search_conversations(q, limit=limit)
    return {"conversations": convs, "count": len(convs)}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    conv = query_db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, limit: int = 200):
    if not DB_ENABLED:
        return {"conversation_id": conversation_id, "messages": [], "count": 0}
    msgs = query_db.get_conversation_messages(conversation_id, limit=limit)
    return {"conversation_id": conversation_id, "messages": msgs, "count": len(msgs)}


@app.put("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameRequest):
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    result = query_db.rename_conversation(conversation_id, req.title)
    if not result:
        raise HTTPException(404, "Conversation not found")
    return result


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    ok = query_db.delete_conversation(conversation_id)
    if not ok:
        raise HTTPException(404, "Conversation not found")
    # Clear in-memory behavior state cache (DB row cascades automatically)
    clear_state(conversation_id)
    return {"deleted": True}


@app.get("/conversations/{conversation_id}/export")
def export_conversation(conversation_id: str):
    """Export a conversation with all messages as JSON."""
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    data = query_db.export_conversation(conversation_id)
    if not data:
        raise HTTPException(404, "Conversation not found")
    return data


@app.get("/conversations/{conversation_id}/state")
def get_conversation_state(conversation_id: str):
    """Inspect behavioral state for a conversation (debug endpoint)."""
    from conversation_state import get_or_create_state as _get_state
    state = _get_state(conversation_id)
    return {
        "conversation_id": conversation_id,
        "state": state.to_dict(),
        "behavior_engine_enabled": settings.BEHAVIOR_ENGINE_ENABLED,
    }


@app.post("/chat/regenerate")
def regenerate_last_response(req: RegenerateRequest):
    """Delete the last assistant message and re-run the pipeline."""
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    msgs = query_db.get_conversation_messages(req.conversation_id, limit=10000)
    user_msgs = [m for m in msgs if m["role"] == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user messages to regenerate from")
    last_user_msg = user_msgs[-1]["content"]
    query_db.delete_last_assistant_message(req.conversation_id)
    return chat(ChatRequest(
        user_query=last_user_msg,
        conversation_id=req.conversation_id,
        user_id=req.user_id,
    ))


# ═══════════════════════════════════════════════════════════════════════════
#  PROFILE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Research engine endpoints ─────────────────────────────────────────────

@app.get("/conversations/{conversation_id}/threads")
def list_threads(conversation_id: str):
    """List all topic threads for a conversation."""
    if not DB_ENABLED:
        return {"threads": [], "count": 0}
    threads = query_db.get_threads(conversation_id)
    # Strip embeddings from response (too large for JSON)
    for t in threads:
        t.pop("centroid_embedding", None)
    return {"threads": threads, "count": len(threads)}


@app.get("/conversations/{conversation_id}/threads/{thread_id}")
def get_thread_detail(conversation_id: str, thread_id: str):
    """Get details of a specific thread including insights."""
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    thread = query_db.get_thread(thread_id)
    if not thread or thread.get("conversation_id") != conversation_id:
        raise HTTPException(404, "Thread not found")
    thread.pop("centroid_embedding", None)
    insights = query_db.get_insights_for_thread(thread_id)
    return {"thread": thread, "insights": insights}


@app.get("/conversations/{conversation_id}/insights")
def list_insights(conversation_id: str, limit: int = 50):
    """List research insights for a conversation."""
    if not DB_ENABLED:
        return {"insights": [], "count": 0}
    insights = query_db.get_insights(conversation_id, limit=limit)
    return {"insights": insights, "count": len(insights)}


@app.get("/conversations/{conversation_id}/concepts")
def list_concepts(conversation_id: str):
    """List concept links for a conversation."""
    if not DB_ENABLED:
        return {"concepts": [], "count": 0}
    concepts = query_db.get_concepts_for_conversation(conversation_id)
    return {"concepts": concepts, "count": len(concepts)}


@app.get("/concepts/search")
def search_concepts(q: str, k: int = 10):
    """Semantic search across all concept links."""
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    embedding = get_query_embedding(q)
    results = query_db.search_similar_concepts(embedding, k=k)
    return {"results": results, "count": len(results)}


@app.get("/insights/search")
def search_insights(
    q: str,
    k: int = 10,
    type: str | None = None,
    conversation_id: str | None = None,
):
    """Cross-thread semantic search over extracted insights.

    Optional filters:
        type — one of: decision, conclusion, hypothesis, open_question, observation
        conversation_id — scope to a single conversation
    """
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    embedding = get_query_embedding(q)
    results = query_db.search_similar_insights(
        embedding, k=k, conversation_id=conversation_id, insight_type=type,
    )
    return {"results": results, "count": len(results)}


# ═══════════════════════════════════════════════════════════════════════════

@app.get("/profile")
def get_profile(user_id: str = settings.DEFAULT_USER_ID):
    if not DB_ENABLED:
        return {"entries": [], "count": 0}
    entries = query_db.get_user_profile(user_id)
    return {"entries": entries, "count": len(entries)}


@app.post("/profile")
def add_profile_entry(req: ProfileEntryRequest):
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    pid = query_db.update_profile_entry(req.key, req.value, category=req.category, user_id=req.user_id)
    if not pid:
        raise HTTPException(500, "Failed to add profile entry")
    return {"id": pid, "key": req.key, "value": req.value, "category": req.category, "user_id": req.user_id}


@app.put("/profile/{entry_id}")
def update_profile(entry_id: int, req: ProfileEntryRequest):
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    pid = query_db.update_profile_entry(req.key, req.value, category=req.category, user_id=req.user_id)
    if not pid:
        raise HTTPException(500, "Failed to update profile entry")
    return {"id": pid, "key": req.key, "value": req.value, "category": req.category, "user_id": req.user_id}


@app.delete("/profile/{entry_id}")
def delete_profile_entry(entry_id: int):
    if not DB_ENABLED:
        raise HTTPException(503, "Database not available")
    ok = query_db.delete_profile_entry(entry_id)
    if not ok:
        raise HTTPException(500, "Failed to delete profile entry")
    return {"deleted": True}


# ═══════════════════════════════════════════════════════════════════════════
#  CHAT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(request: ChatRequest):
    """Non-streaming chat — runs pipeline then returns complete JSON."""
    p = run_pipeline(request)

    try:
        response = generate_response(
            user_query=p.query,
            chat_history=p.recent_messages,
            rag_context=p.rag_context,
            profile_context=p.profile_context,
            similar_qa_context=p.similar_qa_context,
            curated_history=p.curated_history,
            privacy_mode=p.privacy_mode,
            greeting_name=p.greeting_name,
            behavior_context=p.behavior_context,
            meta_instruction=p.meta_instruction,
            personality_mode=p.personality_mode,
            precision_mode=p.precision_mode,
            response_length_hint=p.response_length_hint,
            thread_context=p.thread_context,
            research_context=p.research_context,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {"response": f"Error: {e}", "conversation_id": p.cid}

    persist_after_response(p, str(response))

    return {
        "response": response,
        "conversation_id": p.cid,
        "intent": p.intent,
        "confidence": round(p.confidence, 2),
        "retrieval_info": p.retrieval_info,
        "query_tags": p.query_tags,
        "behavior_mode": p.behavior_mode,
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming chat — Vercel AI SDK data stream protocol over SSE.

    Lines emitted::

        0:"token"\\n          text delta
        8:[{meta}]\\n         metadata annotation
        e:{"finishReason":..}\\n  finish
        d:{"finishReason":..}\\n  done
    """
    try:
        p = run_pipeline(request)
    except Exception as exc:
        logger.error("Pipeline error in chat_stream: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    collected: list[str] = []

    def event_stream():
        # ── Stage events (AI timeline) ────────────────────────────────────
        yield f'8:{json.dumps([{"stage": "classified", "intent": p.intent, "confidence": round(p.confidence, 2)}])}\n'

        ri = p.retrieval_info
        if ri.get("num_docs") or ri.get("similar_queries") or ri.get("same_conv_qa") or ri.get("profile_injected"):
            yield f'8:{json.dumps([{"stage": "retrieved", "retrieval_info": ri}])}\n'

        yield f'8:{json.dumps([{"stage": "generating"}])}\n'

        for chunk in generate_response_stream(
            user_query=p.query,
            chat_history=p.recent_messages,
            rag_context=p.rag_context,
            profile_context=p.profile_context,
            similar_qa_context=p.similar_qa_context,
            curated_history=p.curated_history,
            privacy_mode=p.privacy_mode,
            greeting_name=p.greeting_name,
            behavior_context=p.behavior_context,
            meta_instruction=p.meta_instruction,
            personality_mode=p.personality_mode,
            precision_mode=p.precision_mode,
            response_length_hint=p.response_length_hint,
            thread_context=p.thread_context,
            research_context=p.research_context,
        ):
            if chunk.startswith("0:"):
                try:
                    collected.append(json.loads(chunk[2:].rstrip("\n")))
                except Exception:
                    pass
            yield chunk

        # Emit metadata annotation
        meta = {
            "intent": p.intent,
            "confidence": round(p.confidence, 2),
            "retrieval_info": p.retrieval_info,
            "query_tags": p.query_tags,
            "behavior_mode": p.behavior_mode,
        }
        yield f"8:{json.dumps([meta])}\n"

        # Persist in background
        persist_after_response(p, "".join(collected))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Returns DB status, document count, and provider info."""
    from llm.providers import provider as get_provider

    return {
        "status": "ok",
        "database": "connected" if DB_ENABLED else "unavailable",
        "documents": vector_store.count(),
        "llm_provider": get_provider().name,
        "version": app.version,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SERVE UI
# ═══════════════════════════════════════════════════════════════════════════

# Serve React build assets if available
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.exists() and (_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="frontend-assets")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the React frontend build."""
    react_index = _dist / "index.html"
    if react_index.exists():
        return react_index.read_text(encoding="utf-8")
    return HTMLResponse(
        "<h1>RAG Chat</h1>"
        "<p>Frontend not built. Run <code>cd frontend &amp;&amp; npm run build</code></p>",
        status_code=200,
    )
