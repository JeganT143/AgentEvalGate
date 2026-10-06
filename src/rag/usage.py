"""Token usage and cost metering for every OpenAI call the project makes.

Metered at the HTTP layer: every OpenAI client this project builds (generator, reranker,
embedder, and the RAGAS judge's client) gets an httpx response hook that reads the
`usage` block OpenAI returns with each response. One mechanism covers generation,
reranking, judging and embedding, with no change to how any of them call the SDK.

Usage is attributed with a ContextVar, so concurrent callers (Streamlit sessions,
the full-gate thread pool) each see only their own calls:

    with track_usage() as meter:
        pipeline.answer(query)
    meter.cost_usd

Calls made outside any track_usage() block are simply not recorded.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

# USD per 1M tokens: (input, output). OpenAI list prices - the same rates behind
# results/reranker_benchmark/summary.json's measured cost. Update here if they change.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def price_for(model: str) -> tuple[float, float] | None:
    """Price for a model id, matching dated snapshots ("gpt-4o-mini-2024-07-18") by prefix."""
    matches = [name for name in PRICES_PER_MILLION if model.startswith(name)]
    return PRICES_PER_MILLION[max(matches, key=len)] if matches else None


@dataclass
class UsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    # Models we saw tokens for but have no price for - their cost is NOT in cost_usd.
    unpriced_models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        price = price_for(model)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            if price is None:
                self.unpriced_models.add(model)
            else:
                self.cost_usd += (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_models": sorted(self.unpriced_models),
        }


_CURRENT: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar("aeg_usage_meter", default=None)


@contextlib.contextmanager
def track_usage() -> Iterator[UsageMeter]:
    meter = UsageMeter()
    token = _CURRENT.set(meter)
    try:
        yield meter
    finally:
        _CURRENT.reset(token)


def record_response(response: httpx.Response) -> None:
    """Record the usage block of an (already read) OpenAI JSON response, if metering is on."""
    meter = _CURRENT.get()
    if meter is None or not response.is_success:
        return
    try:
        data = json.loads(response.content)
    except ValueError:
        return
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return
    meter.record(
        str(data.get("model", "unknown")),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _is_json(response: httpx.Response) -> bool:
    # Never read a streamed (text/event-stream) body here - that would consume the stream.
    return response.headers.get("content-type", "").startswith("application/json")


def _on_response(response: httpx.Response) -> None:
    if _CURRENT.get() is not None and _is_json(response):
        response.read()
        record_response(response)


async def _on_response_async(response: httpx.Response) -> None:
    if _CURRENT.get() is not None and _is_json(response):
        await response.aread()
        record_response(response)


def metered_http_client(**kwargs) -> httpx.Client:
    """An httpx client for `OpenAI(http_client=...)` - the SDK's defaults plus the usage hook."""
    from openai import DefaultHttpxClient

    return DefaultHttpxClient(event_hooks={"response": [_on_response]}, **kwargs)


def metered_async_http_client(**kwargs) -> httpx.AsyncClient:
    """The async equivalent, for `AsyncOpenAI(http_client=...)` (the RAGAS judge)."""
    from openai import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(event_hooks={"response": [_on_response_async]}, **kwargs)
