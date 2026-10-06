"""Measure judge score variance across all golden examples, N repeats each.

Standalone diagnostic script, not a pytest test - this is the Day 2 / Step 5
methodology (pin the pipeline output per example, vary only the judge) promoted
from a throwaway local script into something CI can actually invoke. See
internal/mentoring_notes.md (Day 2 / Step 5, Day 3 / Step 5) for the full
methodology and why this needed to move into the repo to run in CI at all.

Usage: python -m src.eval.measure_variance [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.eval.golden_dataset import load_golden_examples
from src.eval.metrics import score_example
from src.rag.factory import build_demo_pipeline
from src.rag.pipeline import RAGPipeline


def process_example(example: dict, pipeline: RAGPipeline, repeats: int) -> dict:
    result = pipeline.answer(example["query"])
    retrieved_context = [r.document.text for r in result.retrieved_context]
    answer = result.answer

    faithfulness_runs = []
    context_precision_runs = []
    for _ in range(repeats):
        scores = score_example(example["query"], retrieved_context, answer)
        faithfulness_runs.append(scores.faithfulness)
        context_precision_runs.append(scores.context_precision)

    return {
        "id": example["id"],
        "category": example["category"],
        "answer": answer,
        "faithfulness_runs": faithfulness_runs,
        "context_precision_runs": context_precision_runs,
    }


def _safe_stdev(values: list[float]) -> float | None:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if len(clean) < 2:
        return None
    return statistics.pstdev(clean)


def summarize(results: list[dict]) -> str:
    lines = []
    ok = [r for r in results if "error" not in r]
    nan_examples = [
        r["id"] for r in ok if any(isinstance(v, float) and math.isnan(v) for v in r["faithfulness_runs"])
    ]
    zero_variance = 0
    nonzero: list[tuple[str, float]] = []
    for r in ok:
        fs = _safe_stdev(r["faithfulness_runs"])
        cs = _safe_stdev(r["context_precision_runs"])
        if fs is not None and fs > 1e-9:
            nonzero.append((r["id"], fs))
        elif fs == 0.0 and (cs is None or cs <= 1e-9):
            zero_variance += 1

    lines.append(f"Total examples: {len(results)}")
    lines.append(f"Examples with >=1 NaN faithfulness run: {len(nan_examples)} -> {nan_examples}")
    lines.append(f"Examples with exactly zero variance (both metrics): {zero_variance}/{len(ok)}")
    lines.append(f"Examples with nonzero faithfulness variance: {nonzero}")

    by_run_faith = defaultdict(list)
    by_run_cp = defaultdict(list)
    for r in ok:
        for i, v in enumerate(r["faithfulness_runs"]):
            if not (isinstance(v, float) and math.isnan(v)):
                by_run_faith[i].append(v)
        for i, v in enumerate(r["context_precision_runs"]):
            by_run_cp[i].append(v)

    faith_means = [statistics.mean(by_run_faith[i]) for i in sorted(by_run_faith)]
    cp_means = [statistics.mean(by_run_cp[i]) for i in sorted(by_run_cp)]
    lines.append(f"Faithfulness mean per run: {[round(m, 4) for m in faith_means]}")
    lines.append(f"Faithfulness stdev across run-means: {statistics.pstdev(faith_means):.6f}")
    lines.append(f"Context precision mean per run: {[round(m, 4) for m in cp_means]}")
    lines.append(f"Context precision stdev across run-means: {statistics.pstdev(cp_means):.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("variance_results.json"))
    args = parser.parse_args()

    # Every golden file, adversarial.jsonl included - the original measurement predates
    # that file and covered v0+v1 only (50 examples); results/judge_variance/summary.json
    # records which population it was measured on.
    examples = load_golden_examples()
    pipeline = build_demo_pipeline()

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_example, ex, pipeline, args.repeats): ex["id"] for ex in examples
        }
        for future in as_completed(futures):
            ex_id = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"id": ex_id, "error": f"{type(e).__name__}: {e}"})
    elapsed = time.time() - start

    results.sort(key=lambda r: r["id"])
    args.out.write_text(json.dumps({"elapsed_seconds": elapsed, "repeats": args.repeats, "results": results}, indent=2))

    summary = f"Judge variance measurement: {len(examples)} examples x {args.repeats} repeats, {elapsed:.1f}s\n\n"
    summary += summarize(results)
    print(summary)

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a") as f:
            f.write("## Judge Variance Measurement\n\n```\n" + summary + "\n```\n")


if __name__ == "__main__":
    main()
