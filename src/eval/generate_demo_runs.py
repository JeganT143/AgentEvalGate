"""Generate real, distinct run results by actually running the real pipeline and
real judge against a small subset of golden examples, varying the generation
prompt across runs to produce genuinely different scores.

Not synthetic data - this is a manual tool for seeding/refreshing real demo data,
e.g. for the dashboard. See internal/mentoring_notes.md, Day 4 / Step 6.

Two modes:
- default: the original 3-run story (baseline -> degraded -> recovered) on 5 typical
  examples, faithfulness only.
- --full-gate: every check the CI gate runs (faithfulness on all `typical` examples,
  declines on every out-of-corpus adversarial example, retrieval on every `multi_hop`
  example), for the grounded prompt and the degraded one, with metered cost. Records each as a RunResult plus its per-question gate report
  (results/gate_reports/), which the dashboard renders as "what a pull request shows".
  Scores with the same functions and policy as tests/eval/ - it's the gate's logic run
  outside pytest, in parallel, so both prompts take a couple of minutes, not twenty.

Usage: python -m src.eval.generate_demo_runs [--full-gate]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from src.config import get_settings
from src.eval.gate_report import GateReport, GateRow
from src.eval.golden_dataset import load_golden_examples
from src.eval.metrics import score_example
from src.eval.policy import (
    FAITHFULNESS_GATED_CATEGORIES,
    FAITHFULNESS_THRESHOLD,
    RETRIEVAL_GATED_CATEGORIES,
    RETRIEVAL_RECALL_BAR,
)
from src.eval.refusal import is_refusal, must_decline
from src.eval.results_store import DEFAULT_GATE_REPORTS_DIR, RunResult, append_run_result
from src.eval.retrieval_metrics import recall_at_k
from src.rag.factory import build_demo_pipeline
from src.rag.pipeline import RAGPipeline
from src.rag.prompts import build_grounded_prompt, build_hallucination_demo_prompt
from src.rag.usage import track_usage

GROUNDED_PROMPT_BUILDER = build_grounded_prompt
# Reuses Day 3 / Step 6's exact hallucination-inducing prompt text (now in
# src/rag/prompts.py), so this script's "degraded" run reproduces that same
# demonstrated regression.
degraded_prompt_builder = build_hallucination_demo_prompt


EXAMPLE_IDS = {"typical-001", "typical-002", "typical-003", "typical-004", "typical-005"}


def load_examples() -> list[dict]:
    return [row for row in load_golden_examples() if row["id"] in EXAMPLE_IDS]


def run_once(run_id: str, prompt_version: str, prompt_builder, examples: list[dict], settings) -> None:
    pipeline = build_demo_pipeline(settings, prompt_builder=prompt_builder)

    faithfulness_scores = []
    context_precision_scores = []
    failing = []
    for example in examples:
        result = pipeline.answer(example["query"])
        retrieved_context = [r.document.text for r in result.retrieved_context]
        scores = score_example(example["query"], retrieved_context, result.answer)
        faithfulness_scores.append(scores.faithfulness)
        context_precision_scores.append(scores.context_precision)
        if scores.faithfulness < 0.5:
            failing.append(
                {
                    "id": example["id"],
                    "category": example["category"],
                    "reason": f"faithfulness {scores.faithfulness:.3f} < threshold 0.5",
                }
            )
        print(f"  {example['id']}: faithfulness={scores.faithfulness:.3f} answer={result.answer!r}")

    faithfulness_mean = statistics.mean(faithfulness_scores)
    context_precision_mean = statistics.mean(context_precision_scores)

    append_run_result(
        RunResult(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dataset_version="v1",
            prompt_version=prompt_version,
            judge_model=settings.judge_model,
            scores={
                "faithfulness_mean": round(faithfulness_mean, 4),
                "context_precision_mean": round(context_precision_mean, 4),
            },
            passed=faithfulness_mean >= 0.5 and not failing,
            failing_examples=failing,
            # Honestly 0.0, not estimated: no code anywhere in this project captures real
            # token usage yet. See internal/build_log.md, Day 4 Summary - a flagged, real gap.
            cost_usd=0.0,
        )
    )
    print(f"-> {run_id}: faithfulness_mean={faithfulness_mean:.4f} failing={len(failing)}")
    time.sleep(2)  # ensure distinct timestamps between runs


def _gate_check(example: dict, pipeline: RAGPipeline) -> GateRow:
    """One gated example, scored exactly as tests/eval/ scores it, with its metered cost."""
    with track_usage() as meter:
        row = _score(example, pipeline)
    return replace(row, cost_usd=meter.cost_usd)


def _score(example: dict, pipeline: RAGPipeline) -> GateRow:
    common = {"example_id": example["id"], "category": example["category"]}
    if example["category"] in RETRIEVAL_GATED_CATEGORIES:
        ids = [r.document.id for r in pipeline.retrieve(example["query"])]
        recall = recall_at_k(ids, example["expected_context_ids"], pipeline.top_k)
        return GateRow(
            **common, check="retrieval", passed=recall >= RETRIEVAL_RECALL_BAR, score=recall,
            threshold=RETRIEVAL_RECALL_BAR, detail=f"top {pipeline.top_k}: {ids}; needed {example['expected_context_ids']}",
        )

    answer_result = pipeline.answer(example["query"])
    if must_decline(example):
        declined = is_refusal(answer_result.answer)
        return GateRow(**common, check="declines", passed=declined, detail=f"answer: {answer_result.answer!r}")

    scores = score_example(
        example["query"], [r.document.text for r in answer_result.retrieved_context], answer_result.answer
    )
    score = scores.faithfulness
    passed = not math.isnan(score) and score >= FAITHFULNESS_THRESHOLD
    return GateRow(
        **common, check="faithfulness", passed=passed, score=score, threshold=FAITHFULNESS_THRESHOLD,
        detail=f"answer: {answer_result.answer!r}", context_precision=scores.context_precision,
    )


def is_gated(example: dict) -> bool:
    return (
        example["category"] in FAITHFULNESS_GATED_CATEGORIES
        or example["category"] in RETRIEVAL_GATED_CATEGORIES
        or must_decline(example)
    )


def run_full_gate(run_id: str, prompt_version: str, prompt_builder, settings, workers: int = 8) -> GateReport:
    pipeline = build_demo_pipeline(settings, prompt_builder=prompt_builder)
    gated = [e for e in load_golden_examples() if is_gated(e)]
    report = GateReport()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(lambda e: _gate_check(e, pipeline), gated):
            report.record(row)
            print(f"  {'pass' if row.passed else 'FAIL'}  {row.example_id} ({row.check})")

    append_run_result(report.to_run_result(run_id, settings.judge_model, prompt_version=prompt_version))
    DEFAULT_GATE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (Path(DEFAULT_GATE_REPORTS_DIR) / f"{run_id}.json").write_text(report.to_json(run_id))
    print(f"-> {run_id}: {report.headline()} cost=${report.cost_usd:.4f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-gate", action="store_true", help="run every gated check, both prompts")
    args = parser.parse_args()

    settings = get_settings()
    run_prefix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    if args.full_gate:
        print("=== Full gate: degraded (hallucination-inducing prompt) ===")
        run_full_gate(f"gate-{run_prefix}-1-degraded", "hallucination-demo", degraded_prompt_builder, settings)
        time.sleep(2)  # ensure distinct timestamps between runs
        print("=== Full gate: current (grounded prompt) ===")
        run_full_gate(f"gate-{run_prefix}-2-grounded", "baseline", GROUNDED_PROMPT_BUILDER, settings)
        return

    examples = load_examples()
    run_prefix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    print("=== Run 1: baseline (grounded prompt) ===")
    run_once(f"demo-{run_prefix}-1-baseline", "baseline", GROUNDED_PROMPT_BUILDER, examples, settings)

    print("=== Run 2: degraded (hallucination-inducing prompt) ===")
    run_once(f"demo-{run_prefix}-2-degraded", "hallucination-demo", degraded_prompt_builder, examples, settings)

    print("=== Run 3: recovered (grounded prompt again) ===")
    run_once(f"demo-{run_prefix}-3-recovered", "baseline", GROUNDED_PROMPT_BUILDER, examples, settings)


if __name__ == "__main__":
    main()
