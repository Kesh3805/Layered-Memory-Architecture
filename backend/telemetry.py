"""Pipeline telemetry — structured instrumentation for every pipeline execution.

Records per-request:
  - Intent classification + confidence
  - Behavior mode + triggers
  - Policy decision (all fields)
  - Thread resolution (attach/create, similarity)
  - Retrieval counts (RAG docs, QA, same-conv, profile)
  - Similarity scores (topic, thread, best-RAG)
  - Token usage (history, prompt estimate, response)
  - Latency per pipeline stage (embed, classify, behavior, thread, research, policy, retrieve, generate)
  - Research memory (insights retrieved, concepts linked)
  - Subsystem activation flags

Usage:
    from telemetry import PipelineTelemetry, TelemetryStore

    t = PipelineTelemetry(conversation_id=cid, query=query)
    t.mark("embed_start")
    ...
    t.mark("embed_end")
    t.record_intent("general", 0.97, source="heuristic:greeting")
    ...
    t.finalize(response_tokens=len(response)//4)
    TelemetryStore.append(t)

    # Export all records
    TelemetryStore.export_jsonl("telemetry_log.jsonl")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE TELEMETRY RECORD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineTelemetry:
    """Single pipeline execution record — captures everything needed to
    evaluate subsystem contribution and measure gate activation."""

    # ── Identity ──────────────────────────────────────────────────────────
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    conversation_id: str = ""
    query: str = ""
    query_word_count: int = 0
    timestamp: float = field(default_factory=time.time)

    # ── Intent classification ─────────────────────────────────────────────
    intent: str = ""
    confidence: float = 0.0
    intent_source: str = ""          # "heuristic:greeting", "heuristic:continuation", "llm", etc.
    intent_overridden: bool = False  # True if topic gate forced general
    original_intent: str = ""        # Pre-override intent

    # ── Topic gate ────────────────────────────────────────────────────────
    topic_similarity: float | None = None
    topic_gate_fired: bool = False   # True if continuation → general

    # ── Behavior engine ───────────────────────────────────────────────────
    behavior_enabled: bool = False
    behavior_mode: str = "standard"
    behavior_triggers: list[str] = field(default_factory=list)
    personality_mode: str = "default"
    precision_mode: str = "analytical"
    response_length_hint: str = "normal"
    emotional_tone: str = "neutral"
    interaction_pattern: str = "normal"
    repetition_count: int = 0
    testing_flag: bool = False

    # ── Topic threading ───────────────────────────────────────────────────
    threading_enabled: bool = False
    thread_id: str = ""
    thread_is_new: bool = False
    thread_similarity: float = 0.0
    thread_label: str = ""
    thread_message_count: int = 0
    total_active_threads: int = 0

    # ── Research memory ───────────────────────────────────────────────────
    research_enabled: bool = False
    insights_retrieved: int = 0
    concepts_retrieved: int = 0
    insights_extracted: int = 0       # Post-response extraction count
    concepts_linked: int = 0          # Post-response linking count

    # ── Policy decision ───────────────────────────────────────────────────
    policy_route: str = ""
    policy_inject_rag: bool = False
    policy_inject_qa: bool = False
    policy_inject_profile: bool = False
    policy_privacy_mode: bool = False
    policy_greeting_name: str = ""
    policy_rag_k: int = 0
    policy_rag_min_similarity: float = 0.0
    policy_qa_k: int = 0

    # ── Behavior overrides on policy ──────────────────────────────────────
    behavior_skip_retrieval: bool = False
    behavior_reduce_retrieval: bool = False
    behavior_boost_retrieval: bool = False

    # ── Retrieval results ─────────────────────────────────────────────────
    rag_docs_retrieved: int = 0
    rag_best_similarity: float = 0.0
    rag_avg_similarity: float = 0.0
    rag_worst_similarity: float = 0.0
    cross_conv_qa_retrieved: int = 0
    same_conv_qa_retrieved: int = 0
    profile_injected: bool = False
    history_raw_count: int = 0
    history_curated_count: int = 0
    history_recency_count: int = 0
    history_semantic_count: int = 0

    # ── Token estimates ───────────────────────────────────────────────────
    query_tokens: int = 0
    history_tokens: int = 0
    rag_tokens: int = 0
    profile_tokens: int = 0
    response_tokens: int = 0
    total_prompt_tokens: int = 0

    # ── Latencies (seconds) ───────────────────────────────────────────────
    latency_embed_ms: float = 0.0
    latency_classify_ms: float = 0.0
    latency_behavior_ms: float = 0.0
    latency_thread_ms: float = 0.0
    latency_research_ms: float = 0.0
    latency_policy_ms: float = 0.0
    latency_history_ms: float = 0.0
    latency_retrieve_ms: float = 0.0
    latency_generate_ms: float = 0.0
    latency_total_ms: float = 0.0

    # ── Stage markers (internal) ──────────────────────────────────────────
    _marks: dict = field(default_factory=dict, repr=False)

    # ── Subsystem activation summary ──────────────────────────────────────
    # Computed at finalize() — handy booleans for analysis
    gate_topic_fired: bool = False
    gate_behavior_frustrated: bool = False
    gate_behavior_testing: bool = False
    gate_behavior_greeting: bool = False
    gate_behavior_rapid_fire: bool = False
    gate_behavior_exploratory: bool = False
    gate_behavior_repetition: bool = False
    gate_thread_attached: bool = False
    gate_thread_created: bool = False
    gate_retrieval_skipped: bool = False
    gate_retrieval_reduced: bool = False
    gate_retrieval_boosted: bool = False

    # ── Methods ───────────────────────────────────────────────────────────

    def mark(self, label: str) -> None:
        """Record a timestamp for latency computation."""
        self._marks[label] = time.perf_counter()

    def _elapsed(self, start: str, end: str) -> float:
        """Milliseconds between two marks."""
        s = self._marks.get(start)
        e = self._marks.get(end)
        if s is not None and e is not None:
            return round((e - s) * 1000, 2)
        return 0.0

    def record_intent(self, intent: str, confidence: float, source: str = "llm") -> None:
        self.intent = intent
        self.confidence = confidence
        self.intent_source = source

    def record_topic_gate(self, similarity: float, fired: bool, original_intent: str = "") -> None:
        self.topic_similarity = round(similarity, 4)
        self.topic_gate_fired = fired
        self.gate_topic_fired = fired
        if fired:
            self.intent_overridden = True
            self.original_intent = original_intent

    def record_behavior(self, decision) -> None:
        """Record from a BehaviorDecision dataclass."""
        self.behavior_enabled = True
        self.behavior_mode = decision.behavior_mode
        self.behavior_triggers = list(decision.triggers) if decision.triggers else []
        self.personality_mode = decision.personality_mode
        self.precision_mode = decision.precision_mode
        self.response_length_hint = decision.response_length_hint
        self.behavior_skip_retrieval = decision.skip_retrieval
        self.behavior_reduce_retrieval = decision.reduce_retrieval
        self.behavior_boost_retrieval = decision.boost_retrieval
        # Gate flags
        self.gate_retrieval_skipped = decision.skip_retrieval
        self.gate_retrieval_reduced = decision.reduce_retrieval
        self.gate_retrieval_boosted = decision.boost_retrieval
        mode = decision.behavior_mode
        if mode == "frustration_recovery":
            self.gate_behavior_frustrated = True
        elif mode == "testing_aware":
            self.gate_behavior_testing = True
        elif mode in ("greeting",):
            self.gate_behavior_greeting = True
        elif mode == "rapid_fire":
            self.gate_behavior_rapid_fire = True
        elif mode == "exploratory":
            self.gate_behavior_exploratory = True
        elif mode == "repetition_aware":
            self.gate_behavior_repetition = True

    def record_state(self, state) -> None:
        """Record from a ConversationState dataclass."""
        self.emotional_tone = state.emotional_tone
        self.interaction_pattern = state.interaction_pattern
        self.repetition_count = state.repetition_count
        self.testing_flag = state.testing_flag

    def record_thread(self, thread_result, total_active: int = 0) -> None:
        """Record from a ThreadResolution result."""
        self.threading_enabled = True
        self.thread_id = thread_result.thread_id or ""
        self.thread_is_new = thread_result.is_new
        self.thread_similarity = round(thread_result.similarity, 4)
        self.thread_label = thread_result.thread_label or ""
        self.thread_message_count = thread_result.message_count
        self.total_active_threads = total_active
        self.gate_thread_attached = not thread_result.is_new and bool(thread_result.thread_id)
        self.gate_thread_created = thread_result.is_new

    def record_research_context(self, research_data: dict | None) -> None:
        self.research_enabled = True
        if research_data:
            self.insights_retrieved = len(research_data.get("related_insights", []))
            self.concepts_retrieved = len(research_data.get("concept_links", []))

    def record_policy(self, decision) -> None:
        """Record from a PolicyDecision object."""
        self.policy_route = decision.retrieval_route
        self.policy_inject_rag = decision.inject_rag
        self.policy_inject_qa = decision.inject_qa_history
        self.policy_inject_profile = decision.inject_profile
        self.policy_privacy_mode = decision.privacy_mode
        self.policy_greeting_name = decision.greeting_name or ""
        self.policy_rag_k = decision.rag_k
        self.policy_rag_min_similarity = decision.rag_min_similarity
        self.policy_qa_k = decision.qa_k

    def record_retrieval(
        self,
        rag_docs: int = 0,
        cross_qa: int = 0,
        same_qa: int = 0,
        profile: bool = False,
        rag_similarities: list[float] | None = None,
    ) -> None:
        self.rag_docs_retrieved = rag_docs
        self.cross_conv_qa_retrieved = cross_qa
        self.same_conv_qa_retrieved = same_qa
        self.profile_injected = profile
        if rag_similarities:
            self.rag_best_similarity = round(max(rag_similarities), 4)
            self.rag_avg_similarity = round(sum(rag_similarities) / len(rag_similarities), 4)
            self.rag_worst_similarity = round(min(rag_similarities), 4)

    def record_history(
        self,
        raw: int = 0,
        curated: int = 0,
        recency: int = 0,
        semantic: int = 0,
    ) -> None:
        self.history_raw_count = raw
        self.history_curated_count = curated
        self.history_recency_count = recency
        self.history_semantic_count = semantic

    def record_tokens(
        self,
        query: int = 0,
        history: int = 0,
        rag: int = 0,
        profile: int = 0,
        response: int = 0,
    ) -> None:
        self.query_tokens = query
        self.history_tokens = history
        self.rag_tokens = rag
        self.profile_tokens = profile
        self.response_tokens = response
        self.total_prompt_tokens = query + history + rag + profile

    def finalize(self) -> None:
        """Compute derived fields from marks."""
        self.query_word_count = len(self.query.split()) if self.query else 0
        self.latency_embed_ms = self._elapsed("embed_start", "embed_end")
        self.latency_classify_ms = self._elapsed("classify_start", "classify_end")
        self.latency_behavior_ms = self._elapsed("behavior_start", "behavior_end")
        self.latency_thread_ms = self._elapsed("thread_start", "thread_end")
        self.latency_research_ms = self._elapsed("research_start", "research_end")
        self.latency_policy_ms = self._elapsed("policy_start", "policy_end")
        self.latency_history_ms = self._elapsed("history_start", "history_end")
        self.latency_retrieve_ms = self._elapsed("retrieve_start", "retrieve_end")
        self.latency_generate_ms = self._elapsed("generate_start", "generate_end")
        self.latency_total_ms = self._elapsed("pipeline_start", "pipeline_end")

    def to_dict(self) -> dict:
        """Serializable dict (excludes internal marks)."""
        d = {}
        for k, v in asdict(self).items():
            if k.startswith("_"):
                continue
            d[k] = v
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  TELEMETRY STORE — in-memory ring buffer + JSONL export
# ═══════════════════════════════════════════════════════════════════════════

class TelemetryStore:
    """Thread-safe in-memory telemetry store with JSONL export.

    Keeps the last ``max_records`` entries in a ring buffer.
    Export to JSONL for offline analysis.
    """

    _records: list[PipelineTelemetry] = []
    _lock = Lock()
    _max_records: int = 10_000

    @classmethod
    def configure(cls, max_records: int = 10_000) -> None:
        cls._max_records = max_records

    @classmethod
    def append(cls, record: PipelineTelemetry) -> None:
        with cls._lock:
            cls._records.append(record)
            if len(cls._records) > cls._max_records:
                cls._records = cls._records[-cls._max_records:]

    @classmethod
    def count(cls) -> int:
        with cls._lock:
            return len(cls._records)

    @classmethod
    def recent(cls, n: int = 20) -> list[dict]:
        with cls._lock:
            return [r.to_dict() for r in cls._records[-n:]]

    @classmethod
    def all_records(cls) -> list[dict]:
        with cls._lock:
            return [r.to_dict() for r in cls._records]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._records.clear()

    @classmethod
    def export_jsonl(cls, path: str | Path) -> int:
        """Write all records to a JSONL file. Returns count written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with cls._lock:
            records = list(cls._records)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.to_json() + "\n")
        logger.info(f"Telemetry: exported {len(records)} records to {path}")
        return len(records)

    @classmethod
    def export_csv(cls, path: str | Path) -> int:
        """Write all records to CSV. Returns count written."""
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with cls._lock:
            records = [r.to_dict() for r in cls._records]
        if not records:
            return 0
        fieldnames = list(records[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                # Flatten lists to comma-separated strings for CSV
                flat = {}
                for k, v in row.items():
                    if isinstance(v, list):
                        flat[k] = ",".join(str(x) for x in v)
                    else:
                        flat[k] = v
                writer.writerow(flat)
        logger.info(f"Telemetry: exported {len(records)} records to {path}")
        return len(records)


    # ── Aggregate queries for analysis ────────────────────────────────────

    @classmethod
    def summary(cls) -> dict:
        """Compute aggregate statistics across all records."""
        with cls._lock:
            records = list(cls._records)
        if not records:
            return {"total_requests": 0}

        total = len(records)
        intents = {}
        behavior_modes = {}
        policy_routes = {}
        gates_fired = {
            "topic_gate": 0,
            "behavior_frustrated": 0,
            "behavior_testing": 0,
            "behavior_greeting": 0,
            "behavior_rapid_fire": 0,
            "behavior_exploratory": 0,
            "behavior_repetition": 0,
            "thread_attached": 0,
            "thread_created": 0,
            "retrieval_skipped": 0,
            "retrieval_reduced": 0,
            "retrieval_boosted": 0,
        }
        latencies = {
            "embed": [], "classify": [], "behavior": [], "thread": [],
            "research": [], "policy": [], "retrieve": [], "generate": [], "total": [],
        }
        subsystem_active = {
            "behavior_engine": 0,
            "threading": 0,
            "research_memory": 0,
            "rag_retrieval": 0,
            "qa_retrieval": 0,
            "profile_injection": 0,
        }
        insight_counts = []
        concept_counts = []
        token_totals = []

        for r in records:
            # Intent distribution
            intents[r.intent] = intents.get(r.intent, 0) + 1
            # Behavior mode distribution
            behavior_modes[r.behavior_mode] = behavior_modes.get(r.behavior_mode, 0) + 1
            # Policy route distribution
            policy_routes[r.policy_route] = policy_routes.get(r.policy_route, 0) + 1

            # Gate activations
            if r.gate_topic_fired:
                gates_fired["topic_gate"] += 1
            if r.gate_behavior_frustrated:
                gates_fired["behavior_frustrated"] += 1
            if r.gate_behavior_testing:
                gates_fired["behavior_testing"] += 1
            if r.gate_behavior_greeting:
                gates_fired["behavior_greeting"] += 1
            if r.gate_behavior_rapid_fire:
                gates_fired["behavior_rapid_fire"] += 1
            if r.gate_behavior_exploratory:
                gates_fired["behavior_exploratory"] += 1
            if r.gate_behavior_repetition:
                gates_fired["behavior_repetition"] += 1
            if r.gate_thread_attached:
                gates_fired["thread_attached"] += 1
            if r.gate_thread_created:
                gates_fired["thread_created"] += 1
            if r.gate_retrieval_skipped:
                gates_fired["retrieval_skipped"] += 1
            if r.gate_retrieval_reduced:
                gates_fired["retrieval_reduced"] += 1
            if r.gate_retrieval_boosted:
                gates_fired["retrieval_boosted"] += 1

            # Latencies
            latencies["embed"].append(r.latency_embed_ms)
            latencies["classify"].append(r.latency_classify_ms)
            latencies["behavior"].append(r.latency_behavior_ms)
            latencies["thread"].append(r.latency_thread_ms)
            latencies["research"].append(r.latency_research_ms)
            latencies["policy"].append(r.latency_policy_ms)
            latencies["retrieve"].append(r.latency_retrieve_ms)
            latencies["generate"].append(r.latency_generate_ms)
            latencies["total"].append(r.latency_total_ms)

            # Subsystem activation
            if r.behavior_enabled:
                subsystem_active["behavior_engine"] += 1
            if r.threading_enabled:
                subsystem_active["threading"] += 1
            if r.research_enabled:
                subsystem_active["research_memory"] += 1
            if r.rag_docs_retrieved > 0:
                subsystem_active["rag_retrieval"] += 1
            if r.cross_conv_qa_retrieved > 0 or r.same_conv_qa_retrieved > 0:
                subsystem_active["qa_retrieval"] += 1
            if r.profile_injected:
                subsystem_active["profile_injection"] += 1

            insight_counts.append(r.insights_retrieved)
            concept_counts.append(r.concepts_retrieved)
            token_totals.append(r.total_prompt_tokens)

        def _stats(values):
            if not values or all(v == 0 for v in values):
                return {"mean": 0, "p50": 0, "p95": 0, "max": 0}
            s = sorted(v for v in values if v > 0) or [0]
            return {
                "mean": round(sum(s) / len(s), 2),
                "p50": round(s[len(s) // 2], 2),
                "p95": round(s[int(len(s) * 0.95)], 2) if len(s) > 1 else round(s[0], 2),
                "max": round(max(s), 2),
            }

        # ── Derived research metrics ──────────────────────────────────────
        rag_best_sims = [r.rag_best_similarity for r in records if r.rag_best_similarity > 0]
        rag_avg_sims = [r.rag_avg_similarity for r in records if r.rag_avg_similarity > 0]

        # Retrieval Precision Proxy: avg best-similarity of injected RAG docs
        retrieval_precision = round(sum(rag_best_sims) / len(rag_best_sims), 4) if rag_best_sims else 0.0

        # Thread Cohesion Score: mean intra-thread similarity (higher = more coherent threads)
        thread_sims = [r.thread_similarity for r in records if r.threading_enabled and r.thread_similarity > 0]
        thread_cohesion = round(sum(thread_sims) / len(thread_sims), 4) if thread_sims else 0.0

        # Thread Fragmentation Rate: % of threaded requests that create new vs attach
        threaded = [r for r in records if r.threading_enabled]
        thread_fragmentation = 0.0
        if threaded:
            new_count = sum(1 for r in threaded if r.thread_is_new)
            thread_fragmentation = round(new_count / len(threaded) * 100, 1)

        # Research Memory ROI: % of requests where insights were surfaced AND query was relevant
        research_requests = [r for r in records if r.research_enabled]
        research_hit_rate = 0.0
        if research_requests:
            hits = sum(1 for r in research_requests if r.insights_retrieved > 0)
            research_hit_rate = round(hits / len(research_requests) * 100, 1)

        # Heuristic Efficiency: % of classifications resolved without LLM
        heuristic_count = sum(1 for r in records if r.intent_source.startswith("heuristic"))
        heuristic_rate = round(heuristic_count / total * 100, 1)

        # Off-topic Injection Rate: % of RAG retrievals with best_similarity < 0.5
        rag_requests = [r for r in records if r.rag_docs_retrieved > 0]
        off_topic_injections = 0.0
        if rag_requests:
            weak = sum(1 for r in rag_requests if r.rag_best_similarity < 0.5)
            off_topic_injections = round(weak / len(rag_requests) * 100, 1)

        # Non-standard behavior rate
        nonstandard_behavior = sum(1 for r in records if r.behavior_mode != "standard")
        nonstandard_rate = round(nonstandard_behavior / total * 100, 1)

        return {
            "total_requests": total,
            "intent_distribution": intents,
            "behavior_mode_distribution": behavior_modes,
            "policy_route_distribution": policy_routes,
            "gate_activations": gates_fired,
            "gate_activation_rates": {
                k: round(v / total * 100, 1) for k, v in gates_fired.items()
            },
            "subsystem_activation": subsystem_active,
            "subsystem_activation_rates": {
                k: round(v / total * 100, 1) for k, v in subsystem_active.items()
            },
            "latency_ms": {k: _stats(v) for k, v in latencies.items()},
            "insights_per_request": _stats(insight_counts),
            "concepts_per_request": _stats(concept_counts),
            "prompt_tokens": _stats(token_totals),
            # ── Derived research metrics ──────────────────────────────────
            "derived_metrics": {
                "retrieval_precision_proxy": retrieval_precision,
                "rag_avg_similarity": _stats(rag_avg_sims),
                "thread_cohesion_score": thread_cohesion,
                "thread_fragmentation_rate": thread_fragmentation,
                "research_memory_hit_rate": research_hit_rate,
                "heuristic_classification_rate": heuristic_rate,
                "off_topic_injection_rate": off_topic_injections,
                "nonstandard_behavior_rate": nonstandard_rate,
            },
        }
