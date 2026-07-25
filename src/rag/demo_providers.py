from __future__ import annotations

import hashlib
import re

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


def hashing_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic, offline placeholder embedder (hashed bag-of-words).

    No semantic understanding - just word overlap. Uses SHA-256 rather than Python's
    built-in hash() because str hashing is randomized per-process by default, which
    would make this "deterministic" placeholder produce different vectors on every
    run. Stand-in until a real embeddings client (Settings.embedding_model_name)
    exists; InMemoryRetriever and RAGPipeline only depend on the Embedder seam, not
    this implementation.

    Tokenizes with a regex (not str.split()) specifically to strip punctuation - a
    query ending in "?" would otherwise hash its last word (e.g. "fastapi?") to a
    completely different bucket than the same word in corpus text ("fastapi"), which
    silently broke retrieval for almost every question-shaped query. See
    internal/mentoring_notes.md, Day 2 / Step 6.
    """
    vector = np.zeros(dim, dtype=np.float64)
    for token in _WORD_RE.findall(text.lower()):
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
