from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_pipeline, limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # The Limiter's in-memory counters are process-global (module-level `limiter` in
    # src/api/main.py), so without a reset, request counts would leak between tests.
    limiter.reset()
    yield
    limiter.reset()


def _stub_pipeline() -> Mock:
    pipeline = Mock()
    pipeline.answer.return_value = Mock(answer="stubbed answer", retrieved_context=[])
    return pipeline


def test_query_allows_ten_requests_per_minute_then_blocks_with_429():
    # No real LLM call happens here - get_pipeline is overridden with a stub, matching
    # the rest of this project's unit tests (see tests/unit/test_pipeline.py).
    app.dependency_overrides[get_pipeline] = _stub_pipeline
    try:
        client = TestClient(app)
        responses = [client.post("/query", json={"query": "test"}) for _ in range(11)]
    finally:
        app.dependency_overrides.clear()

    statuses = [r.status_code for r in responses]
    assert statuses[:10] == [200] * 10, f"expected the first 10 requests to succeed, got {statuses[:10]}"
    assert statuses[10] == 429, f"expected the 11th request within the window to be blocked, got {statuses[10]}"


def test_cors_blocks_an_unlisted_origin_and_allows_the_configured_dashboard_origin():
    client = TestClient(app)

    blocked = client.options(
        "/query",
        headers={"Origin": "http://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in blocked.headers

    allowed = client.options(
        "/query",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
