"""Experiment runner — structured A/B testing for pipeline subsystems.

Runs controlled experiments by toggling subsystems on/off and collecting
telemetry across a fixed set of test queries.

Usage:
    from experiments.runner import ExperimentRunner, Experiment

    runner = ExperimentRunner(base_url="http://localhost:8000")
    results = runner.run_experiment(
        Experiment.CONTINUATION_GATE,
        queries=MULTI_TURN_QUERIES,
    )
    runner.save_results(results, "experiments/results/continuation_gate.json")
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  EXPERIMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

class Experiment(str, Enum):
    """Named experiments — each tests a specific subsystem's contribution."""

    CONTINUATION_GATE = "continuation_gate"
    BEHAVIOR_ENGINE = "behavior_engine"
    THREAD_CLUSTERING = "thread_clustering"
    RESEARCH_MEMORY = "research_memory"
    FULL_PIPELINE = "full_pipeline"
    BASELINE_RAG = "baseline_rag"


@dataclass
class ExperimentConfig:
    """Which subsystems to enable for a given experiment arm."""
    name: str
    behavior_engine: bool = True
    thread_enabled: bool = True
    research_insights: bool = True
    concept_linking: bool = True
    topic_continuation_threshold: float = 0.35
    thread_attach_threshold: float = 0.55
    description: str = ""

    def to_env(self) -> dict:
        return {
            "BEHAVIOR_ENGINE_ENABLED": str(self.behavior_engine).lower(),
            "THREAD_ENABLED": str(self.thread_enabled).lower(),
            "RESEARCH_INSIGHTS_ENABLED": str(self.research_insights).lower(),
            "CONCEPT_LINKING_ENABLED": str(self.concept_linking).lower(),
            "TOPIC_CONTINUATION_THRESHOLD": str(self.topic_continuation_threshold),
            "THREAD_ATTACH_THRESHOLD": str(self.thread_attach_threshold),
        }


# Pre-defined experiment arms
EXPERIMENT_ARMS: dict[Experiment, list[ExperimentConfig]] = {
    Experiment.CONTINUATION_GATE: [
        ExperimentConfig(
            name="with_gate",
            topic_continuation_threshold=0.35,
            description="Topic continuation gate active (default threshold 0.35)",
        ),
        ExperimentConfig(
            name="without_gate",
            topic_continuation_threshold=0.0,
            description="Topic continuation gate disabled (threshold 0.0 — never fires)",
        ),
    ],
    Experiment.BEHAVIOR_ENGINE: [
        ExperimentConfig(
            name="with_behavior",
            behavior_engine=True,
            description="Full behavior engine active",
        ),
        ExperimentConfig(
            name="without_behavior",
            behavior_engine=False,
            description="Behavior engine disabled — all messages use standard mode",
        ),
    ],
    Experiment.THREAD_CLUSTERING: [
        ExperimentConfig(
            name="with_threading",
            thread_enabled=True,
            description="Topic threading active (threshold 0.55)",
        ),
        ExperimentConfig(
            name="without_threading",
            thread_enabled=False,
            description="Topic threading disabled — no thread resolution",
        ),
    ],
    Experiment.RESEARCH_MEMORY: [
        ExperimentConfig(
            name="with_research",
            research_insights=True,
            concept_linking=True,
            description="Research memory active — insights + concepts extracted and retrieved",
        ),
        ExperimentConfig(
            name="without_research",
            research_insights=False,
            concept_linking=False,
            description="Research memory disabled — no insight extraction or concept linking",
        ),
    ],
    Experiment.FULL_PIPELINE: [
        ExperimentConfig(
            name="full_pipeline",
            description="All subsystems active (production config)",
        ),
        ExperimentConfig(
            name="minimal_rag",
            behavior_engine=False,
            thread_enabled=False,
            research_insights=False,
            concept_linking=False,
            description="Minimal RAG baseline — classifier + retrieval only",
        ),
    ],
    Experiment.BASELINE_RAG: [
        ExperimentConfig(
            name="standard_rag",
            behavior_engine=False,
            thread_enabled=False,
            research_insights=False,
            concept_linking=False,
            description="Standard RAG — no behavioral/threading/research layers",
        ),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
#  TEST QUERY SETS
# ═══════════════════════════════════════════════════════════════════════════

# Multi-turn conversations for testing continuation, threading, behavior
MULTI_TURN_QUERIES = [
    # Turn 1: Establish topic
    "What are the main differences between Redis and Memcached for caching?",
    # Turn 2: Follow-up (continuation)
    "Which one would be better for session storage?",
    # Turn 3: Topic shift (should create new thread)
    "How do I set up a PostgreSQL database with Docker?",
    # Turn 4: Follow-up on new topic
    "What about connection pooling?",
    # Turn 5: Cross-thread reference
    "Could Redis work as a session store alongside PostgreSQL?",
    # Turn 6: Greeting (behavior test)
    "Hey thanks for all that info!",
    # Turn 7: Short follow-up
    "Why?",
    # Turn 8: Exploratory
    "What if we used both Redis and PostgreSQL together in a microservices architecture?",
]

REPETITION_QUERIES = [
    "What is machine learning?",
    "What is machine learning?",
    "What is machine learning?",
    "Tell me about machine learning",
]

GREETING_QUERIES = [
    "Hello!",
    "Hey there",
    "Hi, what's up?",
    "How are you",
    "Hello again",
    "Hey hey",
]

BEHAVIORAL_QUERIES = [
    # Frustrated sequence
    "That's not what I asked at all",
    "You keep getting this wrong",
    # Testing sequence
    "Are you ChatGPT?",
    "What model are you?",
    # Rapid-fire
    "Redis?",
    "SQL?",
    "NoSQL?",
    "GraphQL?",
    # Exploratory
    "What if we combined event sourcing with CQRS and used Redis as the write model?",
    "How would that compare to using Kafka with a PostgreSQL read model?",
    "What about adding a GraphQL layer on top?",
]

PROFILE_QUERIES = [
    "My name is Alex and I work at Google",
    "I prefer Python over JavaScript",
    "What's my name?",
    "What do I work with?",
]

ALL_QUERY_SETS = {
    "multi_turn": MULTI_TURN_QUERIES,
    "repetition": REPETITION_QUERIES,
    "greetings": GREETING_QUERIES,
    "behavioral": BEHAVIORAL_QUERIES,
    "profile": PROFILE_QUERIES,
}


# ═══════════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Results from one arm of an experiment."""
    experiment: str
    arm: str
    config: dict
    queries: list[str]
    responses: list[dict] = field(default_factory=list)
    telemetry: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentRunner:
    """Runs experiments against a live backend instance.

    The backend must be running. The runner sends queries via HTTP
    and collects telemetry from the /telemetry endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _clear_telemetry(self):
        self._session.post(f"{self.base_url}/telemetry/clear")

    def _apply_config(self, config: ExperimentConfig):
        """Apply experiment config via the runtime config API."""
        try:
            self._session.post(
                f"{self.base_url}/experiments/config",
                json={
                    "behavior_engine": config.behavior_engine,
                    "thread_enabled": config.thread_enabled,
                    "research_insights": config.research_insights,
                    "concept_linking": config.concept_linking,
                },
                timeout=5,
            )
            logger.info(f"  Applied config: behavior={config.behavior_engine} "
                        f"thread={config.thread_enabled} research={config.research_insights} "
                        f"concepts={config.concept_linking}")
        except Exception as e:
            logger.warning(f"Failed to apply experiment config: {e}")

    def _reset_config(self):
        """Reset runtime config to defaults."""
        try:
            self._session.post(f"{self.base_url}/experiments/reset", timeout=5)
        except Exception:
            pass

    def _get_telemetry(self, n: int = 100) -> list[dict]:
        r = self._session.get(f"{self.base_url}/telemetry/recent", params={"n": n})
        if r.status_code == 200:
            return r.json().get("records", [])
        return []

    def _get_summary(self) -> dict:
        r = self._session.get(f"{self.base_url}/telemetry")
        if r.status_code == 200:
            return r.json()
        return {}

    def _create_conversation(self, title: str = "Experiment") -> str:
        r = self._session.post(
            f"{self.base_url}/conversations",
            json={"title": title},
        )
        return r.json()["id"]

    def _send_message(self, query: str, conversation_id: str) -> dict:
        """Send a chat message and return the full response."""
        start = time.perf_counter()
        try:
            r = self._session.post(
                f"{self.base_url}/chat",
                json={"user_query": query, "conversation_id": conversation_id},
                timeout=120,
            )
            elapsed = (time.perf_counter() - start) * 1000
            if r.status_code == 200:
                data = r.json()
                data["_latency_ms"] = round(elapsed, 2)
                return data
            else:
                return {"error": f"HTTP {r.status_code}", "_latency_ms": round(elapsed, 2)}
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {"error": str(e), "_latency_ms": round(elapsed, 2)}

    def _delete_conversation(self, cid: str):
        try:
            self._session.delete(f"{self.base_url}/conversations/{cid}")
        except Exception:
            pass

    def run_arm(
        self,
        arm_config: ExperimentConfig,
        queries: list[str],
        experiment_name: str,
    ) -> ExperimentResult:
        """Run one experiment arm: create conversation, send queries, collect telemetry."""
        logger.info(f"Running arm: {arm_config.name} ({arm_config.description})")

        # Apply runtime config for this arm
        self._apply_config(arm_config)

        # Clear telemetry before this arm
        self._clear_telemetry()

        # Create a fresh conversation for this arm
        cid = self._create_conversation(f"Experiment: {experiment_name} / {arm_config.name}")

        result = ExperimentResult(
            experiment=experiment_name,
            arm=arm_config.name,
            config=asdict(arm_config),
            queries=list(queries),
        )

        # Send each query sequentially (simulates real conversation)
        for i, query in enumerate(queries):
            logger.info(f"  [{i+1}/{len(queries)}] {query[:60]}...")
            response = self._send_message(query, cid)
            result.responses.append(response)
            result.latencies_ms.append(response.get("_latency_ms", 0))
            if "error" in response:
                result.errors.append(f"Query {i+1}: {response['error']}")
            # Small delay to let background persist complete
            time.sleep(1.5)

        # Collect telemetry
        result.telemetry = self._get_telemetry(n=len(queries) + 10)
        result.summary = self._get_summary()

        # Clean up
        self._delete_conversation(cid)
        self._reset_config()

        return result

    def run_experiment(
        self,
        experiment: Experiment,
        queries: Optional[list[str]] = None,
    ) -> list[ExperimentResult]:
        """Run all arms of an experiment and return results."""
        if not self._health_check():
            raise RuntimeError("Backend not reachable at " + self.base_url)

        arms = EXPERIMENT_ARMS.get(experiment, [])
        if not arms:
            raise ValueError(f"No arms defined for experiment: {experiment}")

        if queries is None:
            queries = MULTI_TURN_QUERIES

        results = []
        for arm in arms:
            result = self.run_arm(arm, queries, experiment.value)
            results.append(result)

        return results

    @staticmethod
    def save_results(results: list[ExperimentResult], path: str | Path) -> None:
        """Save experiment results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(results)} arm results to {path}")

    @staticmethod
    def load_results(path: str | Path) -> list[dict]:
        """Load experiment results from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run experiments from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run pipeline experiments")
    parser.add_argument(
        "experiment",
        choices=[e.value for e in Experiment],
        help="Which experiment to run",
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--queries", choices=list(ALL_QUERY_SETS.keys()), default="multi_turn")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runner = ExperimentRunner(base_url=args.url)
    exp = Experiment(args.experiment)
    queries = ALL_QUERY_SETS[args.queries]

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp.value}")
    print(f"  Queries: {args.queries} ({len(queries)} messages)")
    print(f"  Backend: {args.url}")
    print(f"{'='*60}\n")

    results = runner.run_experiment(exp, queries)

    output = args.output or f"experiments/results/{exp.value}_{int(time.time())}.json"
    ExperimentRunner.save_results(results, output)

    # Print summary
    for result in results:
        print(f"\n--- Arm: {result.arm} ---")
        print(f"  Queries: {len(result.queries)}")
        print(f"  Errors: {len(result.errors)}")
        if result.latencies_ms:
            avg = sum(result.latencies_ms) / len(result.latencies_ms)
            print(f"  Avg latency: {avg:.0f}ms")
        summary = result.summary
        if summary.get("gate_activation_rates"):
            print(f"  Gate activations: {summary['gate_activation_rates']}")
        if summary.get("subsystem_activation_rates"):
            print(f"  Subsystem usage: {summary['subsystem_activation_rates']}")


if __name__ == "__main__":
    main()
