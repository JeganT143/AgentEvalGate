from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

from src.rag.pipeline import Embedder

DEFAULT_CACHE_PATH = Path(".cache/embeddings.json")

logger = logging.getLogger(__name__)


def _preview(text: str, length: int = 40) -> str:
    return text if len(text) <= length else text[:length] + "..."


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
        # One embedder is shared across threads (Streamlit sessions, the full-gate thread
        # pool) - the lock keeps the dict and the on-disk JSON consistent.
        self._lock = threading.Lock()
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
        # The inner embedder's class is part of the key, not just the model name: the eval
        # gate used to wrap the offline HashingEmbedder under the label
        # "text-embedding-3-small", which filed 256-dim hashing vectors under the real
        # model's name - and served them to OpenAIEmbedder once it existed (a shape mismatch
        # at retrieval time). Keying on the implementation too makes that collision impossible
        # and orphans the mislabeled entries.
        namespace = f"{type(self._embedder).__name__}:{self._model_name}"
        return hashlib.sha256(f"{namespace}:{text}".encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        key = self._key(text)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.hits += 1
                logger.info("CACHE HIT  (%s): %r", key[:8], _preview(text))
                return np.array(cached)
            self.misses += 1

        logger.info("CACHE MISS (%s): %r - re-embedding", key[:8], _preview(text))
        vector = self._embedder.embed(text)
        with self._lock:
            self._cache[key] = vector.tolist()
            self._save()
        return vector
