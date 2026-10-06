from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config import get_settings
from src.rag.factory import build_demo_pipeline
from src.rag.pipeline import RAGPipeline

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
    # what build_demo_pipeline() constructs, not RAGPipeline or this endpoint.
    return build_demo_pipeline()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[RetrievedChunk]


@app.get("/health")
def health() -> dict[str, str]:
    # Liveness probe for container platforms (Cloud Run, Container Apps) - deliberately
    # touches no pipeline/LLM code, so it's free, fast, and never rate-limited.
    return {"status": "ok"}


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
            RetrievedChunk(id=r.document.id, text=r.document.text, score=r.score)
            for r in result.retrieved_context
        ],
    )
