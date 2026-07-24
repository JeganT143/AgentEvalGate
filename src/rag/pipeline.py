from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

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
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def answer(self, query: str) -> PipelineResult:
        query_embedding = self._embedder.embed(query)
        retrieved = self._retriever.retrieve(query_embedding, top_k=self._top_k)
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
