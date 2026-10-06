import numpy as np

from src.eval.cache import CachedEmbedder


class CountingEmbedder:
    """Test double: returns a deterministic vector per text, counts real calls."""

    def __init__(self) -> None:
        self.call_count = 0

    def embed(self, text: str) -> np.ndarray:
        self.call_count += 1
        return np.array([float(len(text)), 1.0])


def test_cached_embedder_avoids_recomputing_on_repeat(tmp_path):
    inner = CountingEmbedder()
    cache_path = tmp_path / "embeddings.json"
    cached = CachedEmbedder(inner, model_name="text-embedding-3-small", cache_path=cache_path)

    first = cached.embed("hello world")
    second = cached.embed("hello world")

    assert inner.call_count == 1  # only one real embed call, despite two requests
    assert cached.hits == 1
    assert cached.misses == 1
    assert np.array_equal(first, second)


def test_cached_embedder_isolates_by_model_name(tmp_path):
    inner = CountingEmbedder()
    cache_path = tmp_path / "embeddings.json"
    cache_a = CachedEmbedder(inner, model_name="model-a", cache_path=cache_path)
    cache_b = CachedEmbedder(inner, model_name="model-b", cache_path=cache_path)

    cache_a.embed("same text")
    cache_b.embed("same text")

    # Same text, different model name -> two distinct cache entries, two real embed calls -
    # a text-only cache key would have wrongly returned model-a's cached vector for model-b.
    assert inner.call_count == 2


def test_cached_embedder_persists_across_instances(tmp_path):
    inner = CountingEmbedder()
    cache_path = tmp_path / "embeddings.json"

    first_instance = CachedEmbedder(inner, model_name="text-embedding-3-small", cache_path=cache_path)
    first_instance.embed("persisted text")

    second_instance = CachedEmbedder(inner, model_name="text-embedding-3-small", cache_path=cache_path)
    second_instance.embed("persisted text")

    assert inner.call_count == 1  # second instance loaded the cache file, didn't recompute
    assert second_instance.hits == 1
    assert second_instance.misses == 0


def test_cached_embedder_logs_hit_and_miss(tmp_path, caplog):
    inner = CountingEmbedder()
    cache_path = tmp_path / "embeddings.json"
    cached = CachedEmbedder(inner, model_name="text-embedding-3-small", cache_path=cache_path)

    with caplog.at_level("INFO", logger="src.eval.cache"):
        cached.embed("logged text")  # miss
        cached.embed("logged text")  # hit

    messages = [r.message for r in caplog.records]
    assert any("CACHE MISS" in m and "logged text" in m for m in messages)
    assert any("CACHE HIT" in m and "logged text" in m for m in messages)


class OtherEmbedder(CountingEmbedder):
    """A different embedder implementation, same output shape as CountingEmbedder."""


def test_cached_embedder_isolates_by_embedder_implementation_under_the_same_model_name(tmp_path):
    # Regression: a placeholder embedder cached under a real model's name must never be
    # served to the real embedder (the HashingEmbedder/"text-embedding-3-small" collision).
    cache_path = tmp_path / "embeddings.json"
    placeholder, real = CountingEmbedder(), OtherEmbedder()
    CachedEmbedder(placeholder, model_name="text-embedding-3-small", cache_path=cache_path).embed("same text")
    CachedEmbedder(real, model_name="text-embedding-3-small", cache_path=cache_path).embed("same text")

    assert placeholder.call_count == 1 and real.call_count == 1
