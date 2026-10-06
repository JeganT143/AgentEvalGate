"""The one place the demo RAGPipeline is assembled.

The API, the dashboard, the eval tests and the measurement scripts all used to build
the same embedder -> corpus -> retriever -> generator stack by hand, five copies that
could drift apart. They all call build_demo_pipeline() now - which is also the one
function to replace when pointing AgentEvalGate at your own RAG stack.
"""

from __future__ import annotations

import numpy as np

from src.config import Settings, get_settings
from src.rag.pipeline import Embedder, Generator, PromptBuilder, RAGPipeline
from src.rag.prompts import build_grounded_prompt
from src.rag.retriever import InMemoryRetriever, build_demo_corpus
from src.rag.usage import metered_http_client


def build_embedder(settings: Settings) -> Embedder:
    """The Embedder Settings.embedder selects - real embeddings are always cached."""
    if settings.embedder == "hashing":
        from src.rag.demo_providers import HashingEmbedder

        return HashingEmbedder()

    from src.eval.cache import CachedEmbedder
    from src.rag.embedders import OpenAIEmbedder

    return CachedEmbedder(
        OpenAIEmbedder(api_key=settings.api_key.get_secret_value(), model_name=settings.embedding_model_name),
        model_name=settings.embedding_model_name,
    )


def build_demo_pipeline(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
    reranker: bool = False,
    prompt_builder: PromptBuilder = build_grounded_prompt,
) -> RAGPipeline:
    """Build the demo pipeline over the 8-document demo corpus.

    Defaults: the embedder chosen by Settings.embedder (real OpenAI embeddings behind
    CachedEmbedder, or the offline HashingEmbedder) and a real OpenAIGenerator. Pass
    `embedder`/`generator` to override either (unit tests pass HashingEmbedder +
    StubGenerator). `reranker=True` adds the LLMReranker stage. Settings are only read
    when something actually needs them, so a fully-stubbed pipeline builds with zero
    configuration.
    """
    if embedder is None or generator is None or reranker:
        settings = settings or get_settings()

    embedder = embedder if embedder is not None else build_embedder(settings)
    corpus = build_demo_corpus()
    embeddings = np.stack([embedder.embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)

    if generator is None:
        from src.rag.generators import OpenAIGenerator

        generator = OpenAIGenerator(api_key=settings.api_key.get_secret_value(), model_name=settings.model_name)

    llm_reranker = None
    if reranker:
        from openai import OpenAI

        from src.rag.reranker import LLMReranker

        llm_reranker = LLMReranker(
            client=OpenAI(api_key=settings.api_key.get_secret_value(), http_client=metered_http_client()),
            model_name=settings.model_name,
        )

    return RAGPipeline(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        reranker=llm_reranker,
        prompt_builder=prompt_builder,
    )
