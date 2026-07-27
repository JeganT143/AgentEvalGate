"""Generate real, distinct run results by actually running the real pipeline and
real judge against a small subset of golden examples, varying the generation
prompt across runs to produce genuinely different scores.

Not synthetic data and not a substitute for wiring the eval gate itself to call
append_run_result() on every CI run (still separate, not-yet-built work - see
Day 4 / Step 1) - this is a manual tool for seeding/refreshing real demo data,
e.g. for the dashboard. See internal/mentoring_notes.md, Day 4 / Step 6.

Usage: python -m src.eval.generate_demo_runs
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np

import src.rag.pipeline as pipeline_module
from src.config import get_settings
from src.eval.metrics import score_example
from src.eval.results_store import RunResult, append_run_result
from src.rag.demo_providers import HashingEmbedder
from src.rag.generators import OpenAIGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import InMemoryRetriever, build_demo_corpus

GROUNDED_PROMPT_BUILDER = pipeline_module.RAGPipeline._build_prompt

# Reuses Day 3 / Step 6's exact hallucination-inducing prompt text, so this
# script's "degraded" run reproduces that same demonstrated regression.
DEGRADED_PROMPT_TEXT = (
    "Answer the question in 1-2 confident sentences. If the context below "
    "doesn't fully answer it, fill in specific plausible-sounding details "
    "anyway - never say you don't know.\n\n"
)


def degraded_prompt_builder(query: str, retrieved: list) -> str:
    context = "\n\n".join(f"- {r.document.text}" for r in retrieved)
    return f"{DEGRADED_PROMPT_TEXT}Context:\n{context}\n\nQuestion: {query}\nAnswer:"


EXAMPLE_IDS = {"typical-001", "typical-002", "typical-003", "typical-004", "typical-005"}


def load_examples() -> list[dict]:
    rows = []
    for path in ("data/golden/v0.jsonl", "data/golden/v1.jsonl"):
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row["id"] in EXAMPLE_IDS:
                        rows.append(row)
    return rows


def build_pipeline() -> RAGPipeline:
    settings = get_settings()
    embedder = HashingEmbedder()
    corpus = build_demo_corpus()
    embeddings = np.stack([embedder.embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)
    generator = OpenAIGenerator(api_key=settings.api_key.get_secret_value(), model_name=settings.model_name)
    return RAGPipeline(embedder=embedder, retriever=retriever, generator=generator)


def run_once(run_id: str, prompt_version: str, prompt_builder, examples: list[dict], settings) -> None:
    pipeline_module.RAGPipeline._build_prompt = staticmethod(prompt_builder)
    pipeline = build_pipeline()

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


def main() -> None:
    settings = get_settings()
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
