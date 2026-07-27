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
