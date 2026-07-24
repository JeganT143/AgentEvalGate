from unittest.mock import Mock

import numpy as np

from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Document, InMemoryRetriever


class FixedEmbedder:
    """Test double: always returns the same vector, regardless of input text."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def embed(self, text: str) -> np.ndarray:
        return self._vector


def test_pipeline_mocks_generator_and_builds_prompt_from_retrieved_context():
    documents = [
        Document(id="doc-a", text="FastAPI is a Python web framework."),
        Document(id="doc-b", text="Mount Everest is the tallest mountain."),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    retriever = InMemoryRetriever(documents=documents, embeddings=embeddings)
    embedder = FixedEmbedder(np.array([1.0, 0.0]))  # always matches doc-a
    generator = Mock()
    generator.generate.return_value = "mocked answer"

    pipeline = RAGPipeline(embedder=embedder, retriever=retriever, generator=generator, top_k=1)
    result = pipeline.answer("What framework builds APIs?")

    # No real LLM was called - the generator is a Mock, and we assert against it directly.
    assert result.answer == "mocked answer"
    generator.generate.assert_called_once()

    # The pipeline's own logic (retrieval + prompt-building) is what's under test here:
    # the mocked generator was called with a prompt containing the retrieved context and query.
    prompt_sent_to_generator = generator.generate.call_args[0][0]
    assert "FastAPI is a Python web framework." in prompt_sent_to_generator
    assert "What framework builds APIs?" in prompt_sent_to_generator
    assert result.retrieved_context[0].document.id == "doc-a"
