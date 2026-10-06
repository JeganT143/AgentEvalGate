import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import AsyncOpenAI, OpenAI
from pytest import approx

from src.rag.usage import (
    UsageMeter,
    metered_async_http_client,
    metered_http_client,
    price_for,
    track_usage,
)


def _fake_openai(request: httpx.Request) -> httpx.Response:
    """A stand-in OpenAI API: one chat completion, with a usage block like the real one."""
    body = {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "FastAPI"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
    }
    return httpx.Response(200, json=body)


def _client() -> OpenAI:
    return OpenAI(
        api_key="test", base_url="http://openai.test/v1",
        http_client=metered_http_client(transport=httpx.MockTransport(_fake_openai)),
    )


def _ask(client: OpenAI) -> str:
    return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "q"}]) \
        .choices[0].message.content


def test_prices_match_dated_snapshots_by_prefix():
    assert price_for("gpt-4o-mini-2024-07-18") == price_for("gpt-4o-mini") == (0.15, 0.60)
    assert price_for("some-unknown-model") is None


def test_meter_costs_tokens_and_flags_unpriced_models():
    meter = UsageMeter()
    meter.record("gpt-4o-mini", 1_000_000, 1_000_000)
    meter.record("mystery-model", 10, 10)

    assert meter.cost_usd == approx(0.75)
    assert meter.calls == 2 and meter.prompt_tokens == 1_000_010
    assert meter.unpriced_models == {"mystery-model"}


def test_the_real_sdk_still_parses_the_response_after_the_hook_records_usage():
    with track_usage() as meter:
        answer = _ask(_client())

    assert answer == "FastAPI"
    assert (meter.calls, meter.prompt_tokens, meter.completion_tokens) == (1, 1000, 200)
    assert meter.cost_usd == approx((1000 * 0.15 + 200 * 0.60) / 1_000_000)


def test_async_client_is_metered_too():
    client = AsyncOpenAI(
        api_key="test", base_url="http://openai.test/v1",
        http_client=metered_async_http_client(transport=httpx.MockTransport(_fake_openai)),
    )

    async def ask():
        return await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "q"}])

    with track_usage() as meter:
        asyncio.run(ask())  # how RAGAS's sync .score() drives the judge
    assert meter.completion_tokens == 200


def test_calls_outside_track_usage_are_not_recorded_and_threads_stay_separate():
    client = _client()
    _ask(client)  # no meter active: nothing to record into, nothing breaks

    def one_metered_call(_):
        with track_usage() as meter:
            _ask(client)
        return meter.calls

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(one_metered_call, range(8))) == [1] * 8


def test_error_and_non_json_responses_are_ignored():
    def failing(request):
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = OpenAI(
        api_key="test", base_url="http://openai.test/v1", max_retries=0,
        http_client=metered_http_client(transport=httpx.MockTransport(failing)),
    )
    with track_usage() as meter:
        try:
            _ask(client)
        except Exception:
            pass
    assert meter.calls == 0
    assert json.loads(json.dumps(meter.to_dict()))["cost_usd"] == 0
