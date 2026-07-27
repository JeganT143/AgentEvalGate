from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.rag.reranker import Reranker
from src.rag.retriever import InMemoryRetriever, RetrievalResult


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    retrieved_context: list[RetrievalResult]


class RAGPipeline:
    """Orchestrates retrieve -> build prompt -> generate for a single query.

    Depends only on the Embedder/Retriever/Generator seams, never a concrete
    provider - see internal/mentoring_notes.md (Day 1 / Step 5) for why this is
    a single injectable class rather than a chain of free functions, and how
    it lets providers - or a reranker (Day 5) - be swapped in without
    touching this class or the API layer.
    """

    def __init__(
        self,
        embedder: Embedder,
        retriever: InMemoryRetriever,
        generator: Generator,
        top_k: int = 3,
        reranker: Reranker | None = None,
        rerank_pool_size: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._reranker = reranker
        # Oversample beyond top_k when reranking, so there's room to actually change
        # which documents land in the final top_k - not just their order within an
        # unchanged set, which precision@k can't register at all (see
        # internal/mentoring_notes.md, Day 5 / Step 2's measured finding). Defaults to
        # the same +2 oversample already measured in that step.
        self._rerank_pool_size = rerank_pool_size if rerank_pool_size is not None else top_k + 2

    def answer(self, query: str) -> PipelineResult:
        query_embedding = self._embedder.embed(query)
        retrieve_k = self._rerank_pool_size if self._reranker is not None else self._top_k
        retrieved = self._retriever.retrieve(query_embedding, top_k=retrieve_k)
        if self._reranker is not None:
            retrieved = self._reranker.rerank(query, retrieved, top_k=self._top_k)
        prompt = self._build_prompt(query, retrieved)
        generated_answer = self._generator.generate(prompt)
        return PipelineResult(answer=generated_answer, retrieved_context=retrieved)

    @staticmethod
    def _build_prompt(query: str, retrieved: list[RetrievalResult]) -> str:
        context = "\n\n".join(f"- {result.document.text}" for result in retrieved)
        return (
            "Answer the question using only the context below. "
            "If the context doesn't contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )


if __name__ == "__main__":
    from src.rag.demo_providers import HashingEmbedder, StubGenerator
    from src.rag.retriever import build_demo_corpus

    embedder = HashingEmbedder()
    corpus = build_demo_corpus()
    embeddings = np.stack([embedder.embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)
    pipeline = RAGPipeline(embedder=embedder, retriever=retriever, generator=StubGenerator())

    result = pipeline.answer("What Python framework is used to build web APIs?")
    print("Answer:", result.answer)
    print("\nRetrieved context:")
    for retrieval in result.retrieved_context:
        print(f"  [{retrieval.score:.3f}] {retrieval.document.text}")
