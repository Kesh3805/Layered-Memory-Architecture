# Layered Memory Architecture (LMA) — Architecture & Core Concepts

## What This Is

LMA is a policy-driven, intent-gated, memory-layered conversational AI framework. It is NOT a naive RAG system that dumps all context into every LLM call. Instead, it classifies user intent first, routes through a behavioral intelligence layer, then selectively retrieves only the context that is actually needed. This reduces noise, cuts latency, and produces better responses.

The framework implements four explicit memory tiers, topic threading, research memory with concept linking, and an 8-mode behavior engine — features that commercial assistants like ChatGPT handle implicitly in opaque model weights.

Version: 6.0.0. Backend: FastAPI. Frontend: React 18 + Vite + Tailwind + Vercel AI SDK. Database: PostgreSQL 16 + pgvector (single DB, no FAISS). LLM: Pluggable (Cerebras, OpenAI, Anthropic). Tests: 297 across 13 files.

## Four Memory Tiers

| Tier | Storage | What it stores | Updated |
|---|---|---|---|
| A) Episodic | user_queries table | Facts extracted from queries + embeddings for semantic search | Every message |
| B) Semantic | user_profile table | User traits, preferences, personal facts (key-value) | On profile statement detection |
| C) Conversational | conversation_state table | Behavioral patterns: tone, repetition, testing, interaction pattern, precision mode | Every message |
| D) Research | research_insights + concept_links tables | Decisions, conclusions, hypotheses, concept cross-links across threads | After each response (background) |

## The Five Intent Categories

Every message is classified into one of five intents BEFORE any retrieval happens:

- **general**: Greetings, opinions, open questions, small talk. Minimal retrieval. LLM gets system prompt + query only.
- **knowledge_base**: Factual or technical questions that benefit from document retrieval. LLM gets knowledge base excerpts + prior Q&A + history + query.
- **continuation**: Follow-up messages referencing earlier conversation (uses pronouns like "it", "that", "those"). LLM gets curated history + same-conversation Q&A + query.
- **profile**: Personal information sharing ("My name is Alex") or personal queries ("What's my name?"). Profile updates are saved in background. Profile questions inject the stored profile.
- **privacy**: Questions about what data is stored, deletion requests, tracking concerns. LLM gets full profile data + privacy transparency rules.

## The Pipeline (12 Steps)

Every chat message goes through this exact sequence in run_pipeline() in main.py:

Step 1 — EMBED: Convert user query to 768-dimensional float32 vector using sentence-transformers (BAAI/bge-base-en-v1.5, runs locally). Uses get_query_embedding() with QUERY_INSTRUCTION prefix for asymmetric retrieval.

Step 2 — PARALLEL LOAD: Simultaneously load conversation history and user profile from PostgreSQL. Run in parallel using ThreadPoolExecutor(3) to cut latency from ~90ms to ~50ms.

Step 3 — CLASSIFY INTENT: Pre-heuristics first (greeting, profile statement, privacy, continuation fast-paths). LLM fallback if no heuristic matches. Redis cache check before any work (if enabled).

Step 4 — TOPIC GATE: Only for "continuation" intent. Cosine similarity between query embedding and conversation's rolling topic vector. If below TOPIC_CONTINUATION_THRESHOLD (0.35), demotes to "general" to prevent false continuation across domain jumps.

Step 4b — BEHAVIOR ENGINE: Load/create conversation state from cache (LRU, 200 entries) or DB. StateTracker.update() detects: emotional tone, repetition (Jaccard ≥ 0.7), testing/adversarial behavior, meta-commentary, interaction pattern (rapid_fire/exploratory/deep_dive/standard). BehaviorEngine.evaluate() runs 8 priority-ordered modes → BehaviorDecision (includes precision_mode: concise/analytical/speculative/implementation/adversarial).

Step 4c — THREAD RESOLUTION: If THREAD_ENABLED, resolve_thread() finds nearest thread by cosine_similarity(query_embedding, centroid). Attach if above THREAD_ATTACH_THRESHOLD (0.55), else create new thread (up to THREAD_MAX_ACTIVE=12 per conversation). Thread centroids use EMA with α=0.3.

Step 4d — RESEARCH CONTEXT: If RESEARCH_INSIGHTS_ENABLED, get_research_context() fetches related insights + concept links via semantic search. Packaged into research_context dict for prompt injection.

Step 5 — POLICY RESOLVE: Extract ContextFeatures + BehaviorPolicy.resolve() + behavior overrides from BehaviorDecision + Hooks.run_policy_override().

Step 6 — HISTORY PRUNING: Recency window (last RECENCY_WINDOW=6 messages) + semantic retrieval (top SEMANTIC_K=3 similar older messages). Merge + deduplicate → curated_history.

Step 7 — SELECTIVE RETRIEVAL: Based on PolicyDecision flags: inject_rag → vector search, inject_qa_history → Q&A search, inject_profile → format profile entries, privacy_mode → transparency frame.

Step 8 — BEFORE_GENERATION HOOKS: Run all @Hooks.before_generation functions on PipelineResult.

Step 9 — GENERATE: Build message list via build_messages() in prompt_orchestrator.py (includes behavior_context, personality_mode, response_length_hint, thread_context, research_context). Stream or batch via configured LLM provider.

Step 10 — AFTER_GENERATION HOOKS: Run @Hooks.after_generation and @Hooks.before_persist functions.

Step 11 — BACKGROUND PERSIST: worker.submit(): save messages, save query embedding, detect profile updates, update topic vector (α=TOPIC_DECAY_ALPHA=0.2), auto-title first message, extract insights (LLM-powered), link concepts (heuristic), maybe_summarize thread.

## Six Core Subsystems

### Intent Classification (classifier.py)
5-intent taxonomy with pre-heuristic fast paths (greeting at 0.97, profile statement at 0.92, privacy at 0.95, continuation at 0.85 confidence) before LLM fallback. Redis cache integration. Robust JSON parsing with markdown fence stripping.

### Behavior Policy (policy.py)
Deterministic rules that map intent + ContextFeatures → PolicyDecision. Separates behavior from model-calling code. When behavior is wrong, fix a rule — never edit prompts. Cross-intent overlays handle name injection and personal reference detection.

### Topic Threading (topic_threading.py)
Groups messages into topical threads using EMA centroid similarity. Thread resolution finds/creates threads per message. Thread summaries generated at THREAD_SUMMARY_INTERVAL (8, 16, 24...) milestones via thread_summarizer.py. Labels auto-generated on first summary.

### Research Memory (research_memory.py)
LLM-powered insight extraction (types: decision, conclusion, hypothesis, open_question, observation) with confidence thresholds. Heuristic concept extraction (capitalized nouns, snake_case/camelCase terms, quoted/backtick terms, acronyms). Cross-thread semantic search.

### Behavior Engine (behavior_engine.py)
8 priority-ordered behavior modes: frustration_recovery, testing_aware, meta_aware, repetition_aware, greeting, rapid_fire, exploratory, standard. Modulates retrieval depth, personality mode (default/concise/detailed/playful/empathetic), response length hints (brief/normal/detailed), and injects behavioral context into prompts.

### Conversation State (conversation_state.py)
19-field ConversationState dataclass tracking: current_topic, topic_turns_stable, topic_drift_count, emotional_tone, tone_shift_count, interaction_pattern, testing_flag, repetition_count, meta_comment_count, last_intent, intent_history, intent_streak, message_count, avg_query_length, short_query_streak, dynamic_personality_mode, last_update, conversation_start, precision_mode (5 modes: concise/analytical/speculative/implementation/adversarial).

## Streaming Stage Protocol

Before text tokens, the backend sends three stage annotation events for the frontend:

1. `8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]`
2. `8:[{"stage":"retrieved","retrieval_info":{"num_docs":4,"similar_queries":2}}]`
3. `8:[{"stage":"generating"}]`

Text tokens: `0:"token text here"`. Final annotation with full metadata. Finish: `e:{"finishReason":"stop"}` + `d:{"finishReason":"stop"}`.

## Single Database Architecture

PostgreSQL 16 + pgvector is the single database for everything: 9 tables (conversations, chat_messages, user_queries, user_profile, document_chunks, conversation_state, conversation_threads, research_insights, concept_links). All vector columns use HNSW indexing. No FAISS. No separate vector DB. One connection pool, one backup strategy.

## Topic Vector Rolling Average

Each conversation has a rolling topic embedding updated on every message using EMA: new_vector = (1 - α) * old_vector + α * query_embedding, where α = TOPIC_DECAY_ALPHA (0.2). Recent messages influence the topic more than older ones. Used in Step 4 to detect topic drift.

## Why Policy-Driven Architecture

The BehaviorPolicy separates behavior from model calls. When behavior is wrong (e.g., greeting doesn't use the name), fix a rule in policy.py — never edit prompts or generators. When retrieval is wrong, adjust a policy rule. This makes the system debuggable and extensible via hooks.
