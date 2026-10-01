"""R4: the toolset registers nimble_research and an agent calls it. R5: retries never create a second billable run
unless creation was definitely rejected. R6: the model sees the compact text; the full result is kept on the toolset."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from pydantic_ai_nimble import (
    NimbleAPIError,
    NimbleAuthError,
    NimbleClient,
    NimbleCreateAmbiguousError,
    NimbleSettings,
    ResearchResult,
)
from pydantic_ai_nimble.toolset import NimbleToolset
from tests.conftest import AGENT_ID, RUN_ID, TEST_KEY, fixture_response, route_success

GRADES_HEADER = "Nimble's confidence per reported claim (numbers are the [n] markers in the answer):"


@pytest.fixture
def toolset(client: NimbleClient) -> NimbleToolset:
    return NimbleToolset(client)


def parts(result, kind: str) -> list:
    return [part for message in result.all_messages() for part in message.parts if part.part_kind == kind]


# ------------------------------------------------------------------ R4


def test_toolset_registers_one_tool_with_a_schema_from_the_docstring(toolset: NimbleToolset) -> None:
    assert list(toolset.tools) == ["nimble_research"]
    tool = toolset.tools["nimble_research"]
    assert tool.description and tool.description.startswith("Research a question on the live web")
    schema = tool.function_schema.json_schema
    assert set(schema["properties"]) == {"query", "effort"}
    assert schema["required"] == ["query"]
    assert "plain language" in schema["properties"]["query"]["description"]
    assert set(schema["properties"]["effort"]["anyOf"][0]["enum"]) == {"low", "medium", "high"}


async def test_agent_calls_the_tool_and_gets_the_compact_answer(
    toolset: NimbleToolset, mock_api: respx.MockRouter
) -> None:
    routes = route_success(mock_api)
    agent = Agent(TestModel(), toolsets=[toolset])

    result = await agent.run("What changed in the EU AI Act?")

    assert routes["create"].call_count == 1 and routes["result"].call_count == 1
    sent = json.loads(routes["create"].calls[0].request.content)
    assert sent["input"], "TestModel passed a query string through"
    assert sent["effort"] == "low", "the default effort is applied when the model does not choose"
    returns = [part.content for part in parts(result, "tool-return") if part.tool_name == "nimble_research"]
    assert len(returns) == 1
    assert returns[0].startswith("Answer (Nimble confidence: high")
    assert GRADES_HEADER in returns[0]
    assert toolset.pending == {}


def test_missing_key_fails_at_construction_before_any_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
    with pytest.raises(NimbleAuthError, match="NIMBLE_API_KEY is not set"):
        NimbleToolset()


def test_explicit_key_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
    toolset = NimbleToolset(api_key=TEST_KEY, default_effort="medium")
    assert toolset.client.settings.api_key.get_secret_value() == TEST_KEY
    assert TEST_KEY not in repr(toolset.client)


# ------------------------------------------------------------------ R5: retries never recreate a run


async def test_timeout_then_retry_collects_the_same_run(settings: NimbleSettings, mock_api: respx.MockRouter) -> None:
    """First call times out while the run is active; the model's one retry collects that run, creating nothing."""
    fast = settings.model_copy(update={"deadline_s": 0.05, "poll_initial_s": 0.01, "poll_max_s": 0.01})
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    async with httpx.AsyncClient() as http:
        toolset = NimbleToolset(NimbleClient(fast, http_client=http), max_retries=1)

        def poll(request: httpx.Request) -> httpx.Response:
            # still running until the first attempt has given up and remembered the run; then completed
            return fixture_response("completed") if toolset.pending else fixture_response("running")

        poll_route = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(side_effect=poll)
        result_route = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(
            return_value=fixture_response("result")
        )
        agent = Agent(TestModel(), toolsets=[toolset])

        result = await agent.run("q")

    assert create.call_count == 1, "the retry collected the existing run instead of creating another"
    assert result_route.call_count == 1 and poll_route.call_count >= 2
    retries = parts(result, "retry-prompt")
    assert len(retries) == 1 and f"run {RUN_ID}" in retries[0].model_response()
    assert "same query to collect" in retries[0].model_response()
    returns = [part.content for part in parts(result, "tool-return")]
    assert len(returns) == 1 and returns[0].startswith("Answer (Nimble confidence: high")
    assert toolset.pending == {} and toolset.last_result is not None


async def test_run_still_active_after_the_retry_gives_up_but_keeps_the_handle(
    settings: NimbleSettings, mock_api: respx.MockRouter
) -> None:
    fast = settings.model_copy(update={"deadline_s": 0.05, "poll_initial_s": 0.01, "poll_max_s": 0.01})
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("running"))
    async with httpx.AsyncClient() as http:
        toolset = NimbleToolset(NimbleClient(fast, http_client=http), max_retries=1)
        agent = Agent(TestModel(), toolsets=[toolset])
        with pytest.raises(UnexpectedModelBehavior, match="exceeded max retries count of 1"):
            await agent.run("q")

    assert create.call_count == 1, "two attempts, one run"
    assert list(toolset.pending.values()) == [(AGENT_ID, RUN_ID)], "the developer can still collect it later"


async def test_ambiguous_create_reaches_the_developer_without_a_model_retry(
    toolset: NimbleToolset, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs").mock(side_effect=httpx.ReadTimeout("lost"))
    agent = Agent(TestModel(), toolsets=[toolset])
    with pytest.raises(NimbleCreateAmbiguousError):
        await agent.run("q")
    assert create.call_count == 1 and toolset.pending == {}


async def test_create_rejected_with_5xx_reaches_the_developer(
    toolset: NimbleToolset, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(503, text="upstream"))
    agent = Agent(TestModel(), toolsets=[toolset])
    with pytest.raises(NimbleAPIError) as excinfo:
        await agent.run("q")
    assert excinfo.value.phase == "create" and create.call_count == 1


async def test_create_rate_limited_is_retried_once_because_it_was_definitely_rejected(
    toolset: NimbleToolset, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs").mock(
        side_effect=[httpx.Response(429, text="slow down"), fixture_response("create")]
    )
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))
    agent = Agent(TestModel(), toolsets=[toolset])
    result = await agent.run("q")
    assert create.call_count == 2, "a 429 on creation means nothing was created, so trying again is safe"
    assert len(parts(result, "retry-prompt")) == 1 and toolset.last_result is not None


async def test_result_fetch_failure_then_retry_collects_the_same_run(
    settings: NimbleSettings, mock_api: respx.MockRouter
) -> None:
    no_http_retries = settings.model_copy(update={"max_retries": 0})
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    result_route = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(
        side_effect=[httpx.Response(503, text="upstream"), fixture_response("result")]
    )
    async with httpx.AsyncClient() as http:
        toolset = NimbleToolset(NimbleClient(no_http_retries, http_client=http))
        agent = Agent(TestModel(), toolsets=[toolset])
        result = await agent.run("q")

    assert create.call_count == 1 and result_route.call_count == 2
    retries = parts(result, "retry-prompt")
    assert len(retries) == 1 and "the run continues" in retries[0].model_response()
    assert toolset.last_result is not None and toolset.last_result.confidence == "high"


async def test_auth_failure_is_not_retried_and_reaches_the_developer(
    toolset: NimbleToolset, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("auth_error"))
    agent = Agent(TestModel(), toolsets=[toolset])
    with pytest.raises(NimbleAuthError):
        await agent.run("q")
    assert create.call_count == 1


async def test_client_4xx_is_not_retried(toolset: NimbleToolset, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(422, text="input too long"))
    agent = Agent(TestModel(), toolsets=[toolset])
    with pytest.raises(NimbleAPIError):
        await agent.run("q")
    assert create.call_count == 1


# ------------------------------------------------------------------ R6


async def test_full_result_is_kept_and_the_model_sees_a_bounded_view(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    route_success(mock_api)
    seen: list[ResearchResult] = []
    toolset = NimbleToolset(client, max_sources=3, on_result=seen.append)
    agent = Agent(TestModel(), toolsets=[toolset])

    result = await agent.run("q")

    assert len(seen) == 1 and seen[0] is toolset.last_result
    assert len(seen[0].sources) == 9 and seen[0].trust["confidence"] == "high", "nothing dropped on the client side"
    tool_return = next(part.content for part in parts(result, "tool-return"))
    trust_part = tool_return.split(GRADES_HEADER)[1]
    assert trust_part.count("\n[") == 7, "one graded line per reported claim; nothing renumbered"
