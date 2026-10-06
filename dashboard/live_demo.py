"""Backs the "Try it live" tab: a real golden example through the real pipeline, then
the real pinned judge.

Split into two separately-cached steps - run_pipeline() (search + answer, ~1-3 s) and
judge() (the RAGAS judge, ~10 s) - so the page can show the answer while the judge is
still grading, instead of one opaque 13-second spinner. Both meter their real OpenAI
spend (src/rag/usage.py), so the page can say what a run cost.

Heavy imports (openai via the pipeline factory, ragas/langchain via score_example) are
deferred into the functions that need them: importing this module is cheap, so the
dashboard's first paint doesn't wait on a stack only the Run button uses.
"""

from __future__ import annotations

import time

import streamlit as st

from src.eval.golden_dataset import load_golden_examples
from src.eval.refusal import is_refusal
from src.eval.retrieval_metrics import hit_at_k, precision_at_k, recall_at_k
from src.rag.pipeline import RAGPipeline
from src.rag.prompts import build_grounded_prompt, build_hallucination_demo_prompt
from src.rag.usage import track_usage

# The prompts a visitor can pick between - the "broken" one is the Overview story's
# one-line change, so anyone can watch the gate block it.
PROMPTS = {"normal": build_grounded_prompt, "broken": build_hallucination_demo_prompt}


def get_example(example_id: str) -> dict:
    return next(row for row in load_golden_examples() if row["id"] == example_id)


@st.cache_resource(show_spinner=False)
def get_pipeline(reranker_on: bool, prompt: str) -> RAGPipeline:
    """One pipeline per (reranker, prompt) combination. Cached as a Streamlit *resource*
    (not cache_data): a RAGPipeline holds live client objects, not plain serializable data.
    """
    from src.rag.factory import build_demo_pipeline

    return build_demo_pipeline(reranker=reranker_on, prompt_builder=PROMPTS[prompt])


@st.cache_data(persist="disk", show_spinner=False)
def run_pipeline(example_id: str, reranker_on: bool, prompt: str, embedder: str) -> dict:
    """Search + answer for one golden example, with the retrieval metrics and its cost.

    Cached per (example_id, reranker_on, prompt, embedder): the golden set is a small,
    fixed, known 58 examples, so worst-case real spend is bounded and paid at most once
    per combination, never repeated for the same visitor or the next one - see
    internal/mentoring_notes.md, Day 8 / Step 2, for why this replaces a free-text query
    box (no request-level rate limiter exists on this surface, unlike slowapi on the
    FastAPI side - Day 6 / Step 4). `embedder` is unused in the body; it's part of the
    key so a result computed with one embedder is never served after switching to another.

    Raises on failure (bad/expired key, network error) rather than catching here - the
    caller catches at the call site instead, so a failed attempt is never stored as a
    cached "successful" result that would silently stop retrying.
    """
    example = get_example(example_id)
    pipeline = get_pipeline(reranker_on, prompt)
    with track_usage() as meter:
        result = pipeline.answer(example["query"])

    expected = example["expected_context_ids"]
    retrieved_ids = [r.document.id for r in result.retrieved_context]
    k = pipeline.top_k
    return {
        "answer": result.answer,
        "prompt": result.prompt,
        "timings_ms": result.timings_ms,
        "cost_usd": meter.cost_usd,
        "k": k,
        "retrieved": [
            {
                "rank": rank,
                "id": r.document.id,
                "text": r.document.text,
                "score": r.score,
                "expected": r.document.id in expected,
            }
            for rank, r in enumerate(result.retrieved_context, start=1)
        ],
        "precision_at_k": precision_at_k(retrieved_ids, expected, k),
        "hit_at_k": hit_at_k(retrieved_ids, expected, k),
        "recall_at_k": recall_at_k(retrieved_ids, expected, k),
        "declined": is_refusal(result.answer),
    }


@st.cache_data(persist="disk", show_spinner=False)
def judge(example_id: str, reranker_on: bool, prompt: str, embedder: str) -> dict:
    """Grade run_pipeline()'s answer with the same pinned RAGAS judge the CI gate uses."""
    from src.eval.metrics import score_example

    example = get_example(example_id)
    run = run_pipeline(example_id, reranker_on, prompt, embedder)
    start = time.perf_counter()
    with track_usage() as meter:
        scores = score_example(
            query=example["query"],
            retrieved_context=[chunk["text"] for chunk in run["retrieved"]],
            answer=run["answer"],
        )
    return {
        "faithfulness": scores.faithfulness,
        "context_precision": scores.context_precision,
        "judge_ms": (time.perf_counter() - start) * 1000,
        "cost_usd": meter.cost_usd,
    }
