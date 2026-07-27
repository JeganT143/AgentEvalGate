from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.rag.pipeline import Embedder

DEFAULT_CACHE_PATH = Path(".cache/embeddings.json")


class CachedEmbedder:
    """Embedder wrapper that caches by content hash, so unchanged text is never re-embedded.

    Wraps any Embedder implementation (the same seam RAGPipeline depends on) - adopting
    caching later is a one-line change at a pipeline's construction site, not a retrofit
    across every call site that currently embeds text directly. See
    internal/mentoring_notes.md (Day 3 / Step 3) for the cost failure mode this prevents
    (unmitigated re-embedding cost scaling toward ~$4K/month at high volume, per the
    project plan) and why retrofitting it later would be expensive.

    The cache key includes the embedding model name, not just the text - keying on text
    alone would silently return a stale embedding computed by a different model after a
    model swap, the same class of mistake Day 2 / Step 5 fixed for the judge.
    """

    def __init__(
        self,
        embedder: Embedder,
        model_name: str,
        cache_path: Path = DEFAULT_CACHE_PATH,
    ) -> None:
        self._embedder = embedder
        self._model_name = model_name
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = self._load()
        self.hits = 0
        self.misses = 0

    def _load(self) -> dict[str, list[float]]:
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text())
        return {}

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._model_name}:{text}".encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        key = self._key(text)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return np.array(cached)

        self.misses += 1
        vector = self._embedder.embed(text)
        self._cache[key] = vector.tolist()
        self._save()
        return vector
