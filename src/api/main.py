from functools import lru_cache

import numpy as np
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.config import get_settings
from src.rag.demo_providers import HashingEmbedder
from src.rag.generators import OpenAIGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import InMemoryRetriever, build_demo_corpus

app = FastAPI(title="AgentEvalGate API")


@lru_cache
def get_pipeline() -> RAGPipeline:
    # HashingEmbedder is still today's placeholder Embedder (see src/rag/demo_providers.py) -
    # only the Generator is a real provider call now. Swapping either later means changing
    # what's constructed here, not RAGPipeline or this endpoint.
    settings = get_settings()
    embedder = HashingEmbedder()
    corpus = build_demo_corpus()
    embeddings = np.stack([embedder.embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)
    generator = OpenAIGenerator(api_key=settings.api_key.get_secret_value(), model_name=settings.model_name)
    return RAGPipeline(embedder=embedder, retriever=retriever, generator=generator)


class QueryRequest(BaseModel):
    query: str


class RetrievedChunk(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[RetrievedChunk]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    result = pipeline.answer(request.query)
    return QueryResponse(
        answer=result.answer,
        retrieved_context=[
            RetrievedChunk(text=r.document.text, score=r.score) for r in result.retrieved_context
        ],
    )
