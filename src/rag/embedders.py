from __future__ import annotations

import numpy as np
from openai import OpenAI

from src.rag.usage import metered_http_client


class OpenAIEmbedder:
    """Embedder seam implementation backed by a real OpenAI embeddings call.

    The production replacement for HashingEmbedder's word-overlap placeholder: it
    retrieves by meaning, which is what lets multi-hop questions find both of the
    documents they need. Wrap it in CachedEmbedder (the factory does) so unchanged text
    is never re-embedded. Constructed with an explicit api_key/model_name, matching
    OpenAIGenerator, so it stays a plain seam implementation.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        self._client = OpenAI(api_key=api_key, http_client=metered_http_client())
        self._model_name = model_name

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model_name, input=text)
        return np.array(response.data[0].embedding, dtype=np.float64)
