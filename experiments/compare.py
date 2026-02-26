"""Compare full pipeline vs baseline RAG — produces a markdown table for README.

Usage:
    python experiments/compare.py
    python experiments/compare.py --queries all --output experiments/results/comparison.md

Requires backend running on localhost:8000.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    ALL_QUERY_SETS,
    MULTI_TURN_QUERIES,
)
from experiments.query_corpus import ALL_CONVERSATIONS, SyntheticConversation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON ARMS
# ═══════════════════════════════════════════════════════════════════════════

FULL_PIPELINE = ExperimentConfig(
    name="full_pipeline",
    behavior_engine=True,
    thread_enabled=True,
    research_insights=True,
    concept_linking=True,
    description="All subsystems active",
)

BASELINE_RAG = ExperimentConfig(
    name="baseline_rag",
    behavior_engine=False,
    thread_enabled=False,
    research_insights=False,
    concept_linking=False,
    description="Vanilla RAG — classifier + retrieval only",
)


@dataclass
class ComparisonRow:
    """Single metric comparison between two arms."""
    metric: str
    full: str
    baseline: str
    delta: str
    verdict: str  # ✅ ⚠️ ❌


def _fmt(val, unit="") -> str:
    if isinstance(val, float):
        if unit == "%":
            return f"{val:.1f}%"
        elif unit == "ms":
            return f"{val:.0f}ms"
        else:
            return f"{val:.2f}"
    return str(val)


def _delta(full_val: float, base_val: float, unit: str = "", lower_better: bool = False) -> tuple[str, str]:
    """Compute delta string and verdict emoji."""
    if base_val == 0 and full_val == 0:
        return "—", "—"
    if base_val == 0:
        return "+∞", "⚠️"

    diff = full_val - base_val
    pct = (diff / base_val) * 100 if base_val != 0 else 0

    if unit == "ms":
        sign = "+" if diff >= 0 else ""
        delta_str = f"{sign}{diff:.0f}ms ({sign}{pct:.0f}%)"
    elif unit == "%":
        sign = "+" if diff >= 0 else ""
        delta_str = f"{sign}{diff:.1f}pp"
    else:
        sign = "+" if diff >= 0 else ""
        delta_str = f"{sign}{diff:.2f} ({sign}{pct:.0f}%)"

    # Verdict
    if lower_better:
        verdict = "✅" if diff < 0 else ("⚠️" if diff < base_val * 0.2 else "❌")
    else:
        verdict = "✅" if diff > 0 else ("⚠️" if abs(diff) < base_val * 0.1 else "❌")

    return delta_str, verdict


def build_comparison(full_summary: dict, base_summary: dict, full_result: ExperimentResult, base_result: ExperimentResult) -> list[ComparisonRow]:
    """Build comparison rows from two experiment summaries."""
    rows = []
    dm_full = full_summary.get("derived_metrics", {})
    dm_base = base_summary.get("derived_metrics", {})

    # Latency
    full_lat = full_summary.get("latency_ms", {}).get("total", {}).get("mean", 0)
    base_lat = base_summary.get("latency_ms", {}).get("total", {}).get("mean", 0)
    d, v = _delta(full_lat, base_lat, "ms", lower_better=True)
    rows.append(ComparisonRow("Avg Latency", _fmt(full_lat, "ms"), _fmt(base_lat, "ms"), d, v))

    full_p95 = full_summary.get("latency_ms", {}).get("total", {}).get("p95", 0)
    base_p95 = base_summary.get("latency_ms", {}).get("total", {}).get("p95", 0)
    d, v = _delta(full_p95, base_p95, "ms", lower_better=True)
    rows.append(ComparisonRow("P95 Latency", _fmt(full_p95, "ms"), _fmt(base_p95, "ms"), d, v))

    # Tokens
    full_tok = full_summary.get("prompt_tokens", {}).get("mean", 0)
    base_tok = base_summary.get("prompt_tokens", {}).get("mean", 0)
    d, v = _delta(full_tok, base_tok, "ms", lower_better=True)
    rows.append(ComparisonRow("Avg Prompt Tokens", _fmt(full_tok), _fmt(base_tok), d, v))

    # Retrieval Precision
    full_rp = dm_full.get("retrieval_precision_proxy", 0)
    base_rp = dm_base.get("retrieval_precision_proxy", 0)
    d, v = _delta(full_rp, base_rp)
    rows.append(ComparisonRow("Retrieval Precision", _fmt(full_rp), _fmt(base_rp), d, v))

    # Off-topic injection rate
    full_ot = dm_full.get("off_topic_injection_rate", 0)
    base_ot = dm_base.get("off_topic_injection_rate", 0)
    d, v = _delta(full_ot, base_ot, "%", lower_better=True)
    rows.append(ComparisonRow("Off-Topic Injection Rate", _fmt(full_ot, "%"), _fmt(base_ot, "%"), d, v))

    # Thread Cohesion (only full has it)
    full_tc = dm_full.get("thread_cohesion_score", 0)
    rows.append(ComparisonRow("Thread Cohesion Score", _fmt(full_tc), "N/A", "—", "—"))

    # Thread Fragmentation
    full_tf = dm_full.get("thread_fragmentation_rate", 0)
    rows.append(ComparisonRow("Thread Fragmentation", _fmt(full_tf, "%"), "N/A", "—", "—"))

    # Research Memory Hit Rate
    full_rm = dm_full.get("research_memory_hit_rate", 0)
    rows.append(ComparisonRow("Research Memory Hit Rate", _fmt(full_rm, "%"), "N/A", "—", "—"))

    # Heuristic classification
    full_hc = dm_full.get("heuristic_classification_rate", 0)
    base_hc = dm_base.get("heuristic_classification_rate", 0)
    d, v = _delta(full_hc, base_hc, "%")
    rows.append(ComparisonRow("Heuristic Classification", _fmt(full_hc, "%"), _fmt(base_hc, "%"), d, v))

    # Non-standard behavior
    full_ns = dm_full.get("nonstandard_behavior_rate", 0)
    rows.append(ComparisonRow("Non-Standard Behavior", _fmt(full_ns, "%"), "0.0%", "—", "—"))

    # Errors
    full_err = len(full_result.errors)
    base_err = len(base_result.errors)
    rows.append(ComparisonRow("Errors", str(full_err), str(base_err), "—", "✅" if full_err == 0 else "❌"))

    # Subsystem activations (full only)
    sub_rates = full_summary.get("subsystem_activation_rates", {})
    for sub, rate in sub_rates.items():
        rows.append(ComparisonRow(f"↳ {sub}", _fmt(rate, "%"), "0.0%", "—", "—"))

    return rows


def format_markdown_table(rows: list[ComparisonRow], full_result: ExperimentResult, base_result: ExperimentResult) -> str:
    """Format comparison as a markdown table."""
    lines = []
    lines.append("## Full Pipeline vs Baseline RAG — Comparison\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Queries:** {len(full_result.queries)} turns")
    lines.append(f"**Full pipeline errors:** {len(full_result.errors)} | **Baseline errors:** {len(base_result.errors)}\n")
    lines.append("| Metric | Full Pipeline | Baseline RAG | Delta | |")
    lines.append("|--------|:------------:|:------------:|:-----:|:-:|")
    for row in rows:
        lines.append(f"| {row.metric} | {row.full} | {row.baseline} | {row.delta} | {row.verdict} |")

    lines.append("")
    lines.append("### Key")
    lines.append("- ✅ Full pipeline is better")
    lines.append("- ⚠️ Marginal difference (<20%)")
    lines.append("- ❌ Baseline is better (overhead not justified)")
    lines.append("- — Not applicable for baseline\n")
    return "\n".join(lines)


def format_terminal_table(rows: list[ComparisonRow]) -> str:
    """Format comparison as a fixed-width terminal table."""
    header = f"{'Metric':<30} {'Full Pipeline':>15} {'Baseline RAG':>15} {'Delta':>20} {'':>3}"
    sep = "─" * len(header)
    lines = [sep, header, sep]
    for row in rows:
        lines.append(f"{row.metric:<30} {row.full:>15} {row.baseline:>15} {row.delta:>20} {row.verdict:>3}")
    lines.append(sep)
    return "\n".join(lines)


def run_comparison(
    runner: ExperimentRunner,
    conversations: list[SyntheticConversation] | None = None,
    queries: list[str] | None = None,
) -> tuple[ExperimentResult, ExperimentResult, list[ComparisonRow]]:
    """Run full pipeline and baseline arms, return results and comparison."""

    # Build query list
    if queries is None:
        if conversations:
            queries = []
            for conv in conversations:
                queries.extend(conv.queries)
        else:
            queries = MULTI_TURN_QUERIES

    logger.info(f"Running comparison with {len(queries)} queries...")

    # Arm A: Full pipeline
    logger.info("=" * 60)
    logger.info("  ARM A: Full Pipeline")
    logger.info("=" * 60)
    full_result = runner.run_arm(FULL_PIPELINE, queries, "comparison")

    # Arm B: Baseline RAG
    logger.info("=" * 60)
    logger.info("  ARM B: Baseline RAG")
    logger.info("=" * 60)
    base_result = runner.run_arm(BASELINE_RAG, queries, "comparison")

    # Build comparison
    rows = build_comparison(full_result.summary, base_result.summary, full_result, base_result)

    return full_result, base_result, rows


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline vs Baseline RAG comparison")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument(
        "--queries",
        choices=["multi_turn", "behavioral", "all_corpus", "all_legacy"],
        default="all_corpus",
        help="Which query set to use",
    )
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    runner = ExperimentRunner(base_url=args.url)

    # Select queries
    if args.queries == "all_corpus":
        conversations = ALL_CONVERSATIONS
        queries = None
    elif args.queries == "all_legacy":
        queries = []
        for qs in ALL_QUERY_SETS.values():
            queries.extend(qs)
        conversations = None
    else:
        queries = ALL_QUERY_SETS.get(args.queries, MULTI_TURN_QUERIES)
        conversations = None

    full_result, base_result, rows = run_comparison(runner, conversations, queries)

    # Terminal output
    print("\n" + format_terminal_table(rows))

    # Save markdown
    md_path = Path(args.output or f"experiments/results/comparison_{int(time.time())}.md")
    md_content = format_markdown_table(rows, full_result, base_result)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Saved comparison to {md_path}")

    # Save raw results
    json_path = md_path.with_suffix(".json")
    ExperimentRunner.save_results([full_result, base_result], json_path)
    logger.info(f"Saved raw results to {json_path}")

    # Print derived metrics highlight
    dm = full_result.summary.get("derived_metrics", {})
    if dm:
        print("\n── Derived Metrics (Full Pipeline) ──")
        for k, v in dm.items():
            if isinstance(v, dict):
                print(f"  {k}: mean={v.get('mean', 0)}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
