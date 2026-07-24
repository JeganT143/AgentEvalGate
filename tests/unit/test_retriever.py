import numpy as np
from pytest import approx

from src.rag.retriever import Document, InMemoryRetriever


def test_retriever_ranks_by_cosine_similarity():
    documents = [
        Document(id="aligned", text="points the same direction as the query"),
        Document(id="orthogonal", text="points perpendicular to the query"),
        Document(id="diagonal", text="halfway between aligned and orthogonal"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],  # aligned: cosine similarity to [1, 0] is 1.0
            [0.0, 1.0],  # orthogonal: cosine similarity to [1, 0] is 0.0
            [1.0, 1.0],  # diagonal: cosine similarity to [1, 0] is ~0.7071
        ]
    )
    retriever = InMemoryRetriever(documents=documents, embeddings=embeddings)

    results = retriever.retrieve(query_embedding=np.array([1.0, 0.0]), top_k=3)

    assert [r.document.id for r in results] == ["aligned", "diagonal", "orthogonal"]
    assert results[0].score == approx(1.0)
    assert results[1].score == approx(0.70710678)
    assert results[2].score == approx(0.0, abs=1e-9)


def test_retriever_top_k_clamps_to_corpus_size():
    documents = [Document(id="a", text="first"), Document(id="b", text="second")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    retriever = InMemoryRetriever(documents=documents, embeddings=embeddings)

    results = retriever.retrieve(query_embedding=np.array([1.0, 0.0]), top_k=10)

    assert len(results) == 2
