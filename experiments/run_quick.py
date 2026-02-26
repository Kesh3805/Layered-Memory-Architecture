"""Quick experiment run — 5 core conversations, ~50 queries, reports to stdout."""
import sys
import os
import logging
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

# Add backend to path for runner imports  
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from experiments.runner import ExperimentRunner, ExperimentConfig
from experiments.query_corpus import (
    CONV_DEEP_TECHNICAL,
    CONV_TOPIC_SWITCHING,
    CONV_FRUSTRATION,
    CONV_RAPID_FIRE,
    CONV_REPETITION,
)

# Most diagnostic conversations: deep follow-ups, topic shifts, frustration, rapid fire, repetition
CORE_CONVS = [
    CONV_DEEP_TECHNICAL,   # 11 turns - thread continuity
    CONV_TOPIC_SWITCHING,  # 13 turns - topic gate + threading  
    CONV_FRUSTRATION,      # 10 turns - behavior engine
    CONV_RAPID_FIRE,       # 9 turns  - minimal context queries
    CONV_REPETITION,       # 9 turns  - repetition detection
]

FULL = ExperimentConfig(
    name="full_pipeline",
    behavior_engine=True, thread_enabled=True,
    research_insights=True, concept_linking=True,
    description="All subsystems active",
)

BASELINE = ExperimentConfig(
    name="baseline_rag",
    behavior_engine=False, thread_enabled=False,
    research_insights=False, concept_linking=False,
    description="Vanilla RAG only",
)

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"


def main():
    queries = []
    for c in CORE_CONVS:
        queries.extend(c.queries)
    
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Full Pipeline vs Baseline RAG")
    print(f"  Queries: {len(queries)} turns from {len(CORE_CONVS)} conversations")
    print(f"  Backend: {BASE_URL}")
    print(f"{'='*60}\n")

    runner = ExperimentRunner(base_url=BASE_URL)

    # Arm A: Full Pipeline
    print(f"\n--- ARM A: Full Pipeline ---")
    full_result = runner.run_arm(FULL, queries, "comparison")
    full_summary = full_result.summary

    # Arm B: Baseline
    print(f"\n--- ARM B: Baseline RAG ---")
    base_result = runner.run_arm(BASELINE, queries, "comparison")
    base_summary = base_result.summary

    # === RESULTS ===
    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(queries)} queries per arm")
    print(f"{'='*70}")

    def _get(d, *keys, default=0):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    # Latency
    f_lat = _get(full_summary, "latency_ms", "total", "mean")
    b_lat = _get(base_summary, "latency_ms", "total", "mean")
    f_p95 = _get(full_summary, "latency_ms", "total", "p95")
    b_p95 = _get(base_summary, "latency_ms", "total", "p95")
    print(f"\n  Latency:")
    print(f"    Full:     mean={f_lat:.0f}ms  p95={f_p95:.0f}ms")
    print(f"    Baseline: mean={b_lat:.0f}ms  p95={b_p95:.0f}ms")
    if b_lat > 0:
        print(f"    Delta:    {((f_lat-b_lat)/b_lat*100):+.1f}%")

    # Derived metrics
    f_dm = _get(full_summary, "derived_metrics", default={})
    b_dm = _get(base_summary, "derived_metrics", default={})
    print(f"\n  Derived Metrics:")
    for key in ["retrieval_precision_proxy", "off_topic_injection_rate", 
                "heuristic_classification_rate", "nonstandard_behavior_rate",
                "thread_cohesion_score", "thread_fragmentation_rate",
                "research_memory_hit_rate"]:
        fv = f_dm.get(key, "N/A") if isinstance(f_dm, dict) else "N/A"
        bv = b_dm.get(key, "N/A") if isinstance(b_dm, dict) else "N/A"
        label = key.replace("_", " ").title()
        if isinstance(fv, float) and isinstance(bv, float) and bv != 0:
            delta = ((fv - bv) / bv * 100)
            print(f"    {label:<35} Full={fv:.3f}  Base={bv:.3f}  ({delta:+.1f}%)")
        else:
            print(f"    {label:<35} Full={fv}  Base={bv}")

    # Gate activations (full pipeline only)
    f_gates = _get(full_summary, "gate_activation_rates", default={})
    if isinstance(f_gates, dict) and f_gates:
        print(f"\n  Gate Activations (Full Pipeline):")
        for gate, rate in sorted(f_gates.items(), key=lambda x: -x[1]):
            bar = "#" * int(rate / 2)
            print(f"    {gate:<30} {rate:5.1f}%  {bar}")

    # Subsystem activations
    f_sub = _get(full_summary, "subsystem_activation_rates", default={})
    if isinstance(f_sub, dict) and f_sub:
        print(f"\n  Subsystem Activations (Full Pipeline):")
        for sub, rate in sorted(f_sub.items(), key=lambda x: -x[1]):
            bar = "#" * int(rate / 2)
            print(f"    {sub:<30} {rate:5.1f}%  {bar}")

    # Intent distribution comparison
    f_intent = _get(full_summary, "intent_distribution", default={})
    b_intent = _get(base_summary, "intent_distribution", default={})
    if isinstance(f_intent, dict) and f_intent:
        print(f"\n  Intent Distribution:")
        all_intents = sorted(set(list(f_intent.keys()) + list(b_intent.keys())))
        for intent in all_intents:
            fi = f_intent.get(intent, 0)
            bi = b_intent.get(intent, 0)
            print(f"    {intent:<20} Full={fi:5.1f}%  Base={bi:5.1f}%")

    # Errors
    print(f"\n  Errors:")
    print(f"    Full: {len(full_result.errors)}")
    print(f"    Base: {len(base_result.errors)}")
    if full_result.errors:
        for e in full_result.errors[:5]:
            print(f"      {e}")

    # Save results
    ts = int(time.time())
    results_path = Path(f"experiments/results/quick_comparison_{ts}.json")
    ExperimentRunner.save_results([full_result, base_result], results_path)

    # Save markdown table
    md_lines = [
        "## Full Pipeline vs Baseline RAG\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Queries:** {len(queries)} turns from {len(CORE_CONVS)} conversations\n",
        "| Metric | Full Pipeline | Baseline RAG | Delta |",
        "|--------|:------------:|:------------:|:-----:|",
        f"| Avg Latency | {f_lat:.0f}ms | {b_lat:.0f}ms | {((f_lat-b_lat)/max(b_lat,1)*100):+.0f}% |",
        f"| P95 Latency | {f_p95:.0f}ms | {b_p95:.0f}ms | {((f_p95-b_p95)/max(b_p95,1)*100):+.0f}% |",
    ]
    
    for key, label in [
        ("retrieval_precision_proxy", "Retrieval Precision"),
        ("off_topic_injection_rate", "Off-Topic Injection"),
        ("heuristic_classification_rate", "Heuristic Classification"),
        ("research_memory_hit_rate", "Research Memory Hits"),
        ("thread_cohesion_score", "Thread Cohesion"),
        ("nonstandard_behavior_rate", "Non-Standard Behavior"),
    ]:
        fv = f_dm.get(key, 0) if isinstance(f_dm, dict) else 0
        bv = b_dm.get(key, 0) if isinstance(b_dm, dict) else 0
        if isinstance(fv, float) and isinstance(bv, float) and bv > 0:
            delta = f"{((fv-bv)/bv*100):+.0f}%"
        elif isinstance(fv, float):
            delta = "N/A"
        else:
            delta = "—"
        fv_str = f"{fv:.3f}" if isinstance(fv, float) else str(fv)
        bv_str = f"{bv:.3f}" if isinstance(bv, float) else str(bv)
        md_lines.append(f"| {label} | {fv_str} | {bv_str} | {delta} |")

    md_path = Path(f"experiments/results/comparison_{ts}.md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n  Saved: {results_path}")
    print(f"  Saved: {md_path}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
