"""R4: the toolset registers nimble_research and an agent calls it. R5: one ModelRetry on transient failure, none on
auth. R6: the model sees the compact text; the full result is kept on the toolset."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from pydantic_ai_nimble import NimbleAuthError, NimbleClient, NimbleSettings, ResearchResult
from pydantic_ai_nimble.toolset import NimbleToolset
from tests.conftest import AGENT_ID, RUN_ID, TEST_KEY, fixture_response, route_success


@pytest.fixture
def toolset(client: NimbleClient) -> NimbleToolset:
    return NimbleToolset(client)


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
    tool_returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if part.part_kind == "tool-return" and part.tool_name == "nimble_research"
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].startswith("Answer (confidence: high")
    assert "Sources (numbers match the [n] markers above):" in tool_returns[0]


def test_missing_key_fails_at_construction_before_any_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
    with pytest.raises(NimbleAuthError, match="NIMBLE_API_KEY is not set"):
        NimbleToolset()


def test_explicit_key_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
    toolset = NimbleToolset(api_key=TEST_KEY, default_effort="medium")
    assert toolset.client.settings.api_key.get_secret_value() == TEST_KEY
    assert TEST_KEY not in repr(toolset.client)


# ------------------------------------------------------------------ R5


async def test_timeout_becomes_exactly_one_model_retry(settings: NimbleSettings, mock_api: respx.MockRouter) -> None:
    fast = settings.model_copy(update={"deadline_s": 0.05, "poll_initial_s": 0.01, "poll_max_s": 0.01})
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("running"))
    async with httpx.AsyncClient() as http:
        toolset = NimbleToolset(NimbleClient(fast, http_client=http), max_retries=1)
        agent = Agent(TestModel(), toolsets=[toolset])

        with pytest.raises(UnexpectedModelBehavior, match="exceeded max retries count of 1"):
            await agent.run("q")

    assert create.call_count == 2, "first attempt, one retry prompted by ModelRetry, then the agent gives up"


async def test_transient_5xx_becomes_model_retry_and_the_second_try_succeeds(
    settings: NimbleSettings, mock_api: respx.MockRouter
) -> None:
    no_http_retries = settings.model_copy(update={"max_retries": 0})
    create = mock_api.post("/agents/runs").mock(
        side_effect=[httpx.Response(503, text="upstream"), fixture_response("create")]
    )
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))
    async with httpx.AsyncClient() as http:
        toolset = NimbleToolset(NimbleClient(no_http_retries, http_client=http))
        agent = Agent(TestModel(), toolsets=[toolset])

        result = await agent.run("q")

    assert create.call_count == 2
    retry_prompts = [
        part for message in result.all_messages() for part in message.parts if part.part_kind == "retry-prompt"
    ]
    assert len(retry_prompts) == 1 and "transient" in retry_prompts[0].model_response()
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
    from pydantic_ai_nimble import NimbleAPIError

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
    tool_return = next(
        part.content for message in result.all_messages() for part in message.parts if part.part_kind == "tool-return"
    )
    assert len(tool_return) < 6000
    assert tool_return.count("\n[") == 9, "all nine are cited by the answer, so the cap of 3 does not drop them"
