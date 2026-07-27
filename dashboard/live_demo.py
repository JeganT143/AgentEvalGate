from __future__ import annotations

import time

import numpy as np
import streamlit as st
from openai import OpenAI

from src.config import get_settings
from src.eval.golden_dataset import load_golden_examples
from src.eval.metrics import score_example
from src.eval.retrieval_metrics import precision_at_k
from src.rag.demo_providers import HashingEmbedder
from src.rag.generators import OpenAIGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import LLMReranker
from src.rag.retriever import InMemoryRetriever, build_demo_corpus


@st.cache_resource
def get_pipelines() -> dict[bool, RAGPipeline]:
    """Build the two RAGPipeline variants "Try It Live" toggles between - identical
    except reranker on/off. Cached as a Streamlit *resource* (not cache_data): a
    RAGPipeline holds live client objects, not plain serializable data.
    """
    settings = get_settings()
    embedder = HashingEmbedder()
    corpus = build_demo_corpus()
    embeddings = np.stack([embedder.embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)
    generator = OpenAIGenerator(api_key=settings.api_key.get_secret_value(), model_name=settings.model_name)
    client = OpenAI(api_key=settings.api_key.get_secret_value())
    reranker = LLMReranker(client=client, model_name=settings.model_name)

    return {
        False: RAGPipeline(embedder=embedder, retriever=retriever, generator=generator),
        True: RAGPipeline(embedder=embedder, retriever=retriever, generator=generator, reranker=reranker),
    }


@st.cache_data(persist="disk")
def run_example(example_id: str, reranker_on: bool) -> dict:
    """Run one golden example through the real pipeline + real RAGAS judge, for
    real - the actual product, clickable from the dashboard.

    Cached per (example_id, reranker_on): the golden set is a small, fixed, known
    58 examples x 2 toggle states, so worst-case real spend is bounded and paid
    at most once per combo, never repeated for the same visitor or the next one -
    see internal/mentoring_notes.md, Day 8 / Step 2, for why this replaces a
    free-text query box (no request-level rate limiter exists on this surface,
    unlike slowapi on the FastAPI side - Day 6 / Step 4).

    Raises on failure (bad/expired key, network error, malformed judge output)
    rather than catching here - the caller (dashboard/app.py) catches at the call
    site instead, so a failed attempt is never stored as a cached "successful"
    result that would silently stop retrying.
    """
    example = next(row for row in load_golden_examples() if row["id"] == example_id)
    pipeline = get_pipelines()[reranker_on]

    start = time.perf_counter()
    result = pipeline.answer(example["query"])
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_ids = [r.document.id for r in result.retrieved_context]
    expected_ids = set(example["expected_context_ids"])
    scores = score_example(
        query=example["query"],
        retrieved_context=[r.document.text for r in result.retrieved_context],
        answer=result.answer,
    )

    return {
        "answer": result.answer,
        "retrieved": [
            {
                "id": r.document.id,
                "text": r.document.text,
                "score": r.score,
                "expected": r.document.id in expected_ids,
            }
            for r in result.retrieved_context
        ],
        "precision_at_k": precision_at_k(retrieved_ids, example["expected_context_ids"], k=3),
        "faithfulness": scores.faithfulness,
        "context_precision": scores.context_precision,
        "latency_ms": latency_ms,
    }
