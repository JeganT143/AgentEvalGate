from functools import lru_cache

import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config import get_settings
from src.rag.demo_providers import HashingEmbedder
from src.rag.generators import OpenAIGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import InMemoryRetriever, build_demo_corpus

app = FastAPI(title="AgentEvalGate API")

# CORS: only the known dashboard origin may call this API from a browser. Reading
# Settings here (rather than lazily, like get_pipeline does) means the app now
# requires AEG_API_KEY to even start - see internal/mentoring_notes.md, Day 6 / Step 4,
# for why that tradeoff was accepted, and tests/conftest.py for how unit tests still
# run with zero real configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().dashboard_origin],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Rate limiting: every /query call is a real, metered OpenAI charge with no free-grant
# ceiling of its own (unlike the $0/month Azure hosting story from Day 6 / Step 1) - an
# unauthenticated public endpoint with no cap is unbounded financial exposure to
# anyone who finds the URL. 10/minute/IP bounds a single source's worst-case cost to a
# small, known number instead of an open-ended one. See internal/mentoring_notes.md,
# Day 6 / Step 4, for the real measured cost-per-call behind this choice, and its
# known limitation (in-memory counters, per-replica - not shared across scale-out).
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


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
@limiter.limit("10/minute")
def query(
    request: Request, body: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)
) -> QueryResponse:
    # slowapi's decorator requires a `Request` param (it reads the caller's IP off it via
    # get_remote_address) - hence `body` instead of the old `request` name for the parsed
    # QueryRequest. This doesn't change the wire format: the JSON body's shape is
    # unaffected, only this function's internal parameter names.
    result = pipeline.answer(body.query)
    return QueryResponse(
        answer=result.answer,
        retrieved_context=[
            RetrievedChunk(text=r.document.text, score=r.score) for r in result.retrieved_context
        ],
    )
