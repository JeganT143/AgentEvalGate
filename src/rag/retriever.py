from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float


class InMemoryRetriever:
    """Nearest-neighbor search over a fixed, pre-embedded corpus held in a numpy array.

    No persistence, no external DB - see internal/mentoring_notes.md (Day 1 / Step 4)
    for why, and the concrete scale/persistence triggers for revisiting this.
    """

    def __init__(self, documents: list[Document], embeddings: np.ndarray) -> None:
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({embeddings.shape[0]}) must have the same length"
            )
        self._documents = documents
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normalized = embeddings / norms

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> list[RetrievalResult]:
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / (query_norm if query_norm != 0 else 1.0)
        scores = self._normalized @ normalized_query
        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(document=self._documents[i], score=float(scores[i]))
            for i in top_indices
        ]


def build_demo_corpus() -> list[Document]:
    texts = [
        "Python is a high-level programming language known for readability and simplicity.",
        "FastAPI is a modern Python web framework for building APIs quickly with type hints.",
        "Docker containers package an application with its dependencies for consistent deployment.",
        "Cosine similarity measures the angle between two vectors, ignoring their magnitude.",
        "The mitochondria is the organelle responsible for producing ATP energy in a cell.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "Mount Everest is the tallest mountain above sea level, located in the Himalayas.",
        "The Great Wall of China stretches over 13,000 miles across northern China.",
    ]
    return [Document(id=f"doc-{i}", text=text) for i, text in enumerate(texts)]


if __name__ == "__main__":
    from src.rag.demo_providers import hashing_embed

    corpus = build_demo_corpus()
    embeddings = np.stack([hashing_embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)

    query = "What Python framework is used to build web APIs?"
    results = retriever.retrieve(hashing_embed(query), top_k=3)

    print(f"Query: {query}\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result.score:.3f}] {result.document.text}")
