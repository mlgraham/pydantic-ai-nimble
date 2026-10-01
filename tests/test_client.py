"""R1: create, poll, result. R2: auth, timeout, transient retry. R3: the key never leaks."""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from pydantic_ai_nimble import (
    NimbleAPIError,
    NimbleAuthError,
    NimbleClient,
    NimbleRateLimitError,
    NimbleRunFailedError,
    NimbleSettings,
    NimbleTimeoutError,
    ResearchResult,
)
from tests.conftest import AGENT_ID, RUN_ID, TEST_KEY, fixture_response, load, route_success

# ------------------------------------------------------------------ R1


async def test_success_creates_polls_and_fetches_result(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    routes = route_success(mock_api, running_polls=1)

    result = await client.research("What changed in the EU AI Act?", effort="low")

    assert isinstance(result, ResearchResult)
    assert routes["create"].call_count == 1
    assert routes["poll"].call_count == 2, "one running poll, then the completed one"
    assert routes["result"].call_count == 1
    sent = routes["create"].calls[0].request
    assert sent.headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert b'"input"' in sent.content and b'"effort": "low"' in sent.content.replace(b'":"', b'": "')
    assert result.run.id == RUN_ID and result.run.web_search_agent_id == AGENT_ID
    assert result.run.status == "completed" and result.run.is_active is False
    assert result.answer.startswith("# European Commission")
    assert result.confidence == "high"
    assert len(result.sources) == 9 and len(result.claims) == 7
    assert result.trust == load("result")["body"]["output"]["trust"]


async def test_result_fields_match_the_captured_shape(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    route_success(mock_api)
    result = await client.research("q")
    first = result.sources[0]
    assert first.url.startswith("https://") and first.type in ("primary", "secondary")
    claim = result.claims[0]
    assert claim.citations and claim.citations[0].url.startswith("https://")
    assert claim.citations[0].excerpts is None, "excerpts were null on the captured low-effort run"


async def test_compact_output_keeps_nimble_numbering(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    route_success(mock_api)
    result = await client.research("q")
    text = result.compact()
    assert text.startswith("Answer (confidence: high")
    answer_part, sources_part = text.split("Sources (numbers match the [n] markers above):")
    assert result.cited_markers() == list(range(1, 10)), "the captured answer cites [1] to [9]"
    assert len(result.sources) == 9, "and lists nine sources in that order"
    for marker in result.cited_markers():
        assert f"[{marker}]" in answer_part
        assert f"\n[{marker}] " in sources_part, "every marker resolves to a numbered source"
    assert "[10]" not in sources_part
    assert sources_part.count("https://") == 9
    assert "confidence: high" in sources_part, "claim grades attach to the markers they grade"
    assert '"' not in sources_part, "no excerpt lines when excerpts are null"
    assert len(text) < 6000


async def test_compact_caps_uncited_sources(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    captured = load("result")["body"]
    captured["output"]["content"] = "Only the first source matters [1]."
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=httpx.Response(200, json=captured))
    text = (await client.research("q")).compact(max_sources=3)
    sources_part = text.split("Sources (numbers match the [n] markers above):")[1]
    assert sources_part.count("\n[") == 3 and "[4]" not in sources_part, "uncited sources past the cap are dropped"


async def test_run_failed_status_raises(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    failed = dict(load("completed")["body"], status="failed", error="agent crashed")
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=httpx.Response(200, json=failed))
    with pytest.raises(NimbleRunFailedError, match="agent crashed"):
        await client.research("q")


# ------------------------------------------------------------------ R2


async def test_401_raises_auth_error_without_polling(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("auth_error"))
    poll = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}")

    with pytest.raises(NimbleAuthError) as excinfo:
        await client.research("q")

    assert "401" in str(excinfo.value)
    assert "Invalid API key" in str(excinfo.value), "the plain-text body is surfaced"
    assert create.call_count == 1, "no retry on auth failure"
    assert poll.call_count == 0


async def test_missing_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
    with pytest.raises(NimbleAuthError, match="NIMBLE_API_KEY is not set"):
        NimbleClient()


async def test_deadline_raises_timeout_with_run_id(settings: NimbleSettings, mock_api: respx.MockRouter) -> None:
    fast = settings.model_copy(update={"deadline_s": 0.2, "poll_initial_s": 0.05, "poll_max_s": 0.05})
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    poll = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("running"))
    result = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result")

    async with httpx.AsyncClient() as http:
        with pytest.raises(NimbleTimeoutError) as excinfo:
            await NimbleClient(fast, http_client=http).research("q")

    assert excinfo.value.run_id == RUN_ID
    assert excinfo.value.deadline_s == 0.2
    assert 0.2 <= excinfo.value.elapsed_s < 1.0
    assert poll.call_count >= 1 and result.call_count == 0


async def test_503_is_retried_then_succeeds(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(
        side_effect=[
            httpx.Response(503, text="upstream"),
            httpx.Response(503, text="upstream"),
            fixture_response("create"),
        ]
    )
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))

    result = await client.research("q")

    assert create.call_count == 3
    assert result.confidence == "high"


async def test_503_past_retry_budget_surfaces_api_error(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(503, text="upstream down"))
    with pytest.raises(NimbleAPIError) as excinfo:
        await client.research("q")
    assert excinfo.value.status_code == 503 and "upstream down" in str(excinfo.value)
    assert create.call_count == 4, "one attempt plus three retries"


async def test_429_past_retry_budget_is_rate_limit_error(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    mock_api.post("/agents/runs").mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(NimbleRateLimitError):
        await client.research("q")


async def test_other_4xx_is_not_retried(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(422, json={"detail": "input required"}))
    with pytest.raises(NimbleAPIError) as excinfo:
        await client.research("q")
    assert excinfo.value.status_code == 422 and create.call_count == 1


async def test_transport_error_is_retried(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(side_effect=[httpx.ConnectError("boom"), fixture_response("create")])
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))
    await client.research("q")
    assert create.call_count == 2


# ------------------------------------------------------------------ R3


async def test_key_never_appears_in_repr_logs_or_errors(
    client: NimbleClient, mock_api: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="pydantic_ai_nimble")
    mock_api.post("/agents/runs").mock(return_value=fixture_response("auth_error"))

    with pytest.raises(NimbleAuthError) as excinfo:
        await client.research("q")

    assert TEST_KEY not in repr(client)
    assert TEST_KEY not in repr(client.settings)
    assert TEST_KEY not in str(client.settings)
    assert TEST_KEY not in str(excinfo.value)
    assert TEST_KEY not in caplog.text


def test_settings_reads_env_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMBLE_API_KEY", "nk_from_env")
    monkeypatch.setenv("NIMBLE_DEADLINE_S", "45")
    settings = NimbleSettings.from_env()
    assert settings.api_key.get_secret_value() == "nk_from_env"
    assert settings.deadline_s == 45.0
    assert "nk_from_env" not in repr(settings)
    explicit = NimbleSettings.from_env(api_key="nk_explicit")
    assert explicit.api_key.get_secret_value() == "nk_explicit", "an explicit key wins over the environment"
