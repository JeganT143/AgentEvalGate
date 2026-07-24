from __future__ import annotations

import hashlib

import numpy as np


def hashing_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic, offline placeholder embedder (hashed bag-of-words).

    No semantic understanding - just word overlap. Uses SHA-256 rather than Python's
    built-in hash() because str hashing is randomized per-process by default, which
    would make this "deterministic" placeholder produce different vectors on every
    run. Stand-in until a real embeddings client (Settings.embedding_model_name)
    exists; InMemoryRetriever and RAGPipeline only depend on the Embedder seam, not
    this implementation.
    """
    vector = np.zeros(dim, dtype=np.float64)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dim
        vector[index] += 1.0
    return vector


class HashingEmbedder:
    """Embedder seam implementation backed by hashing_embed - see its docstring."""

    def embed(self, text: str) -> np.ndarray:
        return hashing_embed(text)


class StubGenerator:
    """Generator seam implementation that makes no real LLM call.

    Stands in for a real provider-backed Generator so RAGPipeline's orchestration
    can be exercised and unit-tested today without an API key, network access, or
    cost - and later doubles as the test double for mocking the LLM call boundary
    in pipeline unit tests (per the project's testing strategy).
    """

    def generate(self, prompt: str) -> str:
        return "[stub answer - no real LLM called]"
