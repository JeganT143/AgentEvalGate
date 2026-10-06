from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.rag.prompts import build_grounded_prompt
from src.rag.reranker import Reranker
from src.rag.retriever import RetrievalResult, Retriever


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


PromptBuilder = Callable[[str, list[RetrievalResult]], str]


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    retrieved_context: list[RetrievalResult]
    # The exact prompt sent to the generator, and wall-clock time per stage - both
    # surfaced by the dashboard so a reader can see what the gate actually graded.
    prompt: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """Orchestrates retrieve -> build prompt -> generate for a single query.

    Depends only on the Embedder/Retriever/Generator seams, never a concrete
    provider - see internal/mentoring_notes.md (Day 1 / Step 5) for why this is
    a single injectable class rather than a chain of free functions, and how
    it lets providers - or a reranker (Day 5) - be swapped in without
    touching this class or the API layer.

    `prompt_builder` is a seam too: it's how a prompt change (the thing the eval gate
    exists to catch regressions in) is expressed, without patching this class.
    """

    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 3,
        reranker: Reranker | None = None,
        rerank_pool_size: int | None = None,
        prompt_builder: PromptBuilder = build_grounded_prompt,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        # Oversample beyond top_k when reranking, so there's room to actually change
        # which documents land in the final top_k - not just their order within an
        # unchanged set, which precision@k can't register at all (see
        # internal/mentoring_notes.md, Day 5 / Step 2's measured finding). Defaults to
        # the same +2 oversample already measured in that step.
        self._rerank_pool_size = rerank_pool_size if rerank_pool_size is not None else top_k + 2

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Search (and rerank, if configured) without generating - all a retrieval-only
        check needs, at none of the generation cost."""
        return self._retrieve(query, {})

    def _retrieve(self, query: str, timings: dict[str, float]) -> list[RetrievalResult]:
        start = time.perf_counter()
        query_embedding = self._embedder.embed(query)
        retrieve_k = self._rerank_pool_size if self._reranker is not None else self._top_k
        retrieved = self._retriever.retrieve(query_embedding, top_k=retrieve_k)
        timings["retrieve"] = (time.perf_counter() - start) * 1000

        if self._reranker is not None:
            start = time.perf_counter()
            retrieved = self._reranker.rerank(query, retrieved, top_k=self._top_k)
            timings["rerank"] = (time.perf_counter() - start) * 1000
        return retrieved

    def answer(self, query: str) -> PipelineResult:
        timings: dict[str, float] = {}
        retrieved = self._retrieve(query, timings)
        prompt = self._prompt_builder(query, retrieved)
        start = time.perf_counter()
        generated_answer = self._generator.generate(prompt)
        timings["generate"] = (time.perf_counter() - start) * 1000

        return PipelineResult(
            answer=generated_answer, retrieved_context=retrieved, prompt=prompt, timings_ms=timings
        )


if __name__ == "__main__":
    from src.rag.demo_providers import HashingEmbedder, StubGenerator
    from src.rag.factory import build_demo_pipeline

    pipeline = build_demo_pipeline(embedder=HashingEmbedder(), generator=StubGenerator())

    result = pipeline.answer("What Python framework is used to build web APIs?")
    print("Answer:", result.answer)
    print("\nRetrieved context:")
    for retrieval in result.retrieved_context:
        print(f"  [{retrieval.score:.3f}] {retrieval.document.text}")
