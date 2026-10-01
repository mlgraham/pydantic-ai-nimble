"""Shared fixtures. Every HTTP call is mocked with respx from JSON captured on a real run. No test touches the
network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pydantic_ai_nimble import NimbleClient, NimbleSettings

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://sdk.nimbleway.com/v2"
AGENT_ID = "wsa_90c3bd22d6c047de934183870880b37f"
RUN_ID = "task_run_a6b22e6eeac743c8b45cb8a8da25e203"
TEST_KEY = "nk_test_key_not_real"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def fixture_response(name: str) -> httpx.Response:
    captured = load(name)
    body = captured["body"]
    if isinstance(body, str):
        return httpx.Response(captured["status_code"], text=body)
    return httpx.Response(captured["status_code"], json=body)


@pytest.fixture
def settings() -> NimbleSettings:
    """Fast settings: no real sleeping, 2 s overall deadline."""
    return NimbleSettings(
        api_key=TEST_KEY,
        deadline_s=2.0,
        poll_initial_s=0.001,
        poll_max_s=0.002,
        retry_backoff_s=0.0,
        max_retries=3,
    )


@pytest.fixture
def mock_api():
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


@pytest.fixture
async def client(settings: NimbleSettings, mock_api: respx.MockRouter) -> NimbleClient:
    async with httpx.AsyncClient() as http:
        yield NimbleClient(settings, http_client=http)


def route_success(router: respx.MockRouter, *, running_polls: int = 1) -> dict[str, respx.Route]:
    """Create -> N running polls -> completed -> result, all from fixtures."""
    poll_responses = [fixture_response("running")] * running_polls + [fixture_response("completed")]
    return {
        "create": router.post("/agents/runs").mock(return_value=fixture_response("create")),
        "poll": router.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(side_effect=poll_responses),
        "result": router.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result")),
    }
