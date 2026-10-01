"""R1: create, poll, result. R2: auth, deadline, create-once, read retries. R3: the key never leaks."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
import pytest
import respx
from pydantic import ValidationError

from pydantic_ai_nimble import (
    NimbleAPIError,
    NimbleAuthError,
    NimbleClient,
    NimbleCreateAmbiguousError,
    NimbleRateLimitError,
    NimbleRunFailedError,
    NimbleSettings,
    NimbleTimeoutError,
    ResearchResult,
)
from tests.conftest import AGENT_ID, BASE, RUN_ID, TEST_KEY, fixture_response, load, route_success

GRADES_HEADER = "Nimble's confidence per reported claim (numbers are the [n] markers in the answer):"


def route_result(router: respx.MockRouter, body: dict) -> None:
    router.post("/agents/runs").mock(return_value=fixture_response("create"))
    router.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    router.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=httpx.Response(200, json=body))


def body_of(text: str) -> str:
    """Everything after the header line, which embeds the elapsed time."""
    return text.split("\n", 1)[1]


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
    assert json.loads(sent.content) == {
        "input": "What changed in the EU AI Act?",
        "effort": "low",
        "use_case": "research",
    }
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


async def test_compact_keeps_the_answer_and_grades_claims_by_callout(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    route_success(mock_api)
    result = await client.research("q")
    text = result.compact()

    # The answer ends with its own numbered source index; it is kept verbatim and nothing is renumbered.
    index = result.answer_index()
    assert sorted(index) == list(range(1, 10)), "the captured answer indexes [1] to [9] itself"
    assert result.answer.strip() in text
    answer_part, trust_part = text.split(GRADES_HEADER)

    # Exact identity: every graded claim's citation is the page the answer's own index gives for that marker.
    graded = result.graded_claims()
    assert [claim.callout for claim in graded] == [1, 3, 4, 5, 7, 8, 9], "callouts 2 and 6 are absent, not invented"
    for claim in graded:
        assert claim.citations[0].url == index[claim.callout]
        assert f"\n[{claim.callout}] {claim.confidence}\n    " in trust_part
        assert f"    {claim.citations[0].title}: {claim.citations[0].url}" in trust_part
    assert "\n[2] " not in trust_part and "\n[6] " not in trust_part
    assert "different page" not in trust_part, "the captured claims agree with the answer's index"
    assert trust_part.count("(+1 more citation)") == 2, "claims 1 and 5 carry two citations"
    assert '"' not in trust_part, "no excerpt lines when excerpts are null"


async def test_compact_ignores_the_order_of_the_source_inventory(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    """trust.sources is not in the answer's order (measured); the compact view must not depend on it."""
    captured = load("result")["body"]
    routes = route_success(mock_api)
    baseline = body_of((await client.research("q")).compact())
    captured["output"]["trust"]["sources"] = list(reversed(captured["output"]["trust"]["sources"]))
    routes["poll"].mock(return_value=fixture_response("completed"))
    routes["result"].mock(return_value=httpx.Response(200, json=captured))
    shuffled = body_of((await client.research("q")).compact())
    assert shuffled == baseline


async def test_compact_flags_a_citation_that_disagrees_with_the_answer(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    captured = load("result")["body"]
    claim = next(c for c in captured["output"]["trust"]["claims"] if c["callout"] == 7)
    claim["citations"][0]["url"] = "https://example.com/somewhere-else"
    route_result(mock_api, captured)
    text = (await client.research("q")).compact()
    assert "(note: the answer's own [7] entry is a different page: https://regulations.ai/" in text


async def test_compact_keeps_a_low_grade_that_has_no_citation(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    """Nimble grades an unsupported statement 'low' with no citations; that warning must reach the model."""
    captured = load("result")["body"]
    captured["output"]["trust"]["claims"].append(
        {
            "callout": 2,
            "confidence": "low",
            "reasoning": "No usable citation found for this statement.",
            "citations": [],
        }
    )
    route_result(mock_api, captured)
    text = (await client.research("q")).compact()
    trust_part = text.split(GRADES_HEADER)[1]
    assert "\n[2] low\n    no supporting citation in the trust report: No usable citation found" in trust_part
    assert "\n[6] " not in trust_part, "a marker with no reported claim is still not invented"
    assert "https://" not in trust_part.split("\n[2] low")[1].split("\n[3]")[0], "no url is fabricated for it"


async def test_compact_includes_an_excerpt_when_nimble_returns_one(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    captured = load("result")["body"]
    claim = next(c for c in captured["output"]["trust"]["claims"] if c["callout"] == 1)
    claim["citations"][0]["excerpts"] = ["  The Commission   issued guidelines " + "x" * 300]
    route_result(mock_api, captured)
    text = (await client.research("q")).compact(max_excerpt_chars=60)
    quoted = [line for line in text.splitlines() if line.strip().startswith('"')]
    assert (
        len(quoted) == 1
        and quoted[0].strip().startswith('"The Commission issued guidelines')
        and len(quoted[0].strip()) <= 62
    )


async def test_compact_without_claims_lists_an_inventory_not_a_bibliography(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    captured = load("result")["body"]
    captured["output"]["trust"]["claims"] = []
    route_result(mock_api, captured)
    result = await client.research("q")
    positional = result.compact(3, 60)
    named = result.compact(max_sources=3, max_excerpt_chars=60)
    assert body_of(positional) == body_of(named), "positional order is max_sources, max_excerpt_chars, as in 0.1.0"
    inventory = positional.split("Pages Nimble consulted (an inventory, not the answer's numbering; first 3):")[1]
    assert inventory.count("https://") == 3
    assert "[1]" not in inventory, "no numbers are assigned to the inventory"


async def test_run_failed_status_raises(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    failed = dict(load("completed")["body"], status="failed", error="agent crashed")
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=httpx.Response(200, json=failed))
    with pytest.raises(NimbleRunFailedError, match="agent crashed") as excinfo:
        await client.research("q")
    assert excinfo.value.run_id == RUN_ID and excinfo.value.web_search_agent_id == AGENT_ID


# ------------------------------------------------------------------ R2: auth and deadline


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


def test_blank_key_is_rejected_on_direct_construction() -> None:
    with pytest.raises(ValidationError, match="NIMBLE_API_KEY is not set"):
        NimbleSettings(api_key="   ")


async def test_deadline_raises_timeout_with_run_id(settings: NimbleSettings, mock_api: respx.MockRouter) -> None:
    fast = settings.model_copy(update={"deadline_s": 0.2, "poll_initial_s": 0.05, "poll_max_s": 0.05})
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    poll = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("running"))
    result = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result")

    async with httpx.AsyncClient() as http:
        with pytest.raises(NimbleTimeoutError) as excinfo:
            await NimbleClient(fast, http_client=http).research("q")

    assert excinfo.value.run_id == RUN_ID and excinfo.value.web_search_agent_id == AGENT_ID
    assert excinfo.value.deadline_s == 0.2
    assert 0.2 <= excinfo.value.elapsed_s < 1.0
    assert poll.call_count >= 1 and result.call_count == 0


async def test_timeout_during_result_fetch_keeps_run_id_and_the_per_call_deadline(
    settings: NimbleSettings, mock_api: respx.MockRouter
) -> None:
    """A deadline override, a known run, and a retry sleep that must not overrun the budget."""
    slow_retry = settings.model_copy(update={"retry_backoff_s": 0.5, "max_retries": 3})
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    result = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(
        return_value=httpx.Response(503, text="busy")
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(NimbleTimeoutError) as excinfo:
            await NimbleClient(slow_retry, http_client=http).research("q", deadline_s=0.3)

    error = excinfo.value
    assert error.run_id == RUN_ID and error.web_search_agent_id == AGENT_ID
    assert error.deadline_s == 0.3, "the per-call override, not the settings default"
    assert 0.3 <= error.elapsed_s < 0.6, "elapsed is real, and the 0.5 s retry sleep was clamped to the budget"
    assert result.call_count >= 1


async def test_slowly_streamed_result_body_cannot_outlive_the_deadline(settings: NimbleSettings) -> None:
    """The deadline is a wall-clock boundary around the whole call, not a per-chunk read timeout."""
    result_bytes = json.dumps(load("result")["body"]).encode()

    async def trickle():
        for start in range(0, len(result_bytes), 1500):
            await asyncio.sleep(0.05)  # each chunk is well inside any read timeout; the whole body is not
            yield result_bytes[start : start + 1500]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return fixture_response("create")
        if request.url.path.endswith("/result"):
            return httpx.Response(200, content=trickle(), headers={"content-type": "application/json"})
        return fixture_response("completed")

    started = time.monotonic()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE) as http:
        with pytest.raises(NimbleTimeoutError) as excinfo:
            await NimbleClient(settings, http_client=http).research("q", deadline_s=0.15)
    elapsed = time.monotonic() - started

    assert excinfo.value.run_id == RUN_ID and excinfo.value.deadline_s == 0.15
    assert elapsed < 0.4, f"the body needed about 0.5 s; the call stopped at {elapsed:.2f} s"


async def test_caller_cancellation_is_not_reported_as_a_timeout(
    settings: NimbleSettings, mock_api: respx.MockRouter
) -> None:
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("running"))
    async with httpx.AsyncClient() as http:
        task = asyncio.create_task(NimbleClient(settings, http_client=http).research("q"))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ------------------------------------------------------------------ R2: creation happens once; reads retry


async def test_create_with_no_response_is_ambiguous_and_not_resent(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs").mock(side_effect=httpx.ReadTimeout("no response"))
    with pytest.raises(NimbleCreateAmbiguousError, match="may already be running"):
        await client.research("q")
    assert create.call_count == 1, "Nimble documents creation as billable and non-idempotent: never replayed"


async def test_create_answered_with_5xx_is_definite_and_not_retried(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(503, text="upstream down"))
    with pytest.raises(NimbleAPIError) as excinfo:
        await client.research("q")
    assert excinfo.value.status_code == 503 and excinfo.value.phase == "create"
    assert create.call_count == 1


async def test_create_rate_limited_is_a_definite_rejection(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(NimbleRateLimitError) as excinfo:
        await client.research("q")
    assert excinfo.value.phase == "create" and create.call_count == 1


async def test_poll_5xx_is_retried_then_succeeds(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    poll = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(
        side_effect=[httpx.Response(503, text="x"), httpx.Response(502, text="y"), fixture_response("completed")]
    )
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))

    result = await client.research("q")

    assert create.call_count == 1 and poll.call_count == 3
    assert result.confidence == "high"


async def test_result_5xx_past_retry_budget_surfaces_the_run_identity(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(return_value=fixture_response("completed"))
    result = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(
        return_value=httpx.Response(503, text="down")
    )
    with pytest.raises(NimbleAPIError) as excinfo:
        await client.research("q")
    error = excinfo.value
    assert error.status_code == 503 and error.phase == "result"
    assert error.run_id == RUN_ID and error.web_search_agent_id == AGENT_ID
    assert result.call_count == 4, "one attempt plus three retries"


async def test_transport_error_on_a_read_is_retried(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    mock_api.post("/agents/runs").mock(return_value=fixture_response("create"))
    poll = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(
        side_effect=[httpx.ConnectError("boom"), fixture_response("completed")]
    )
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))
    await client.research("q")
    assert poll.call_count == 2


async def test_other_4xx_is_not_retried(client: NimbleClient, mock_api: respx.MockRouter) -> None:
    create = mock_api.post("/agents/runs").mock(return_value=httpx.Response(422, json={"detail": "input required"}))
    with pytest.raises(NimbleAPIError) as excinfo:
        await client.research("q")
    assert excinfo.value.status_code == 422 and create.call_count == 1


async def test_collect_resumes_an_existing_run_without_creating(
    client: NimbleClient, mock_api: respx.MockRouter
) -> None:
    create = mock_api.post("/agents/runs")
    poll = mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}").mock(
        side_effect=[fixture_response("running"), fixture_response("completed")]
    )
    mock_api.get(f"/agents/{AGENT_ID}/runs/{RUN_ID}/result").mock(return_value=fixture_response("result"))

    result = await client.collect(AGENT_ID, RUN_ID)

    assert create.call_count == 0 and poll.call_count == 2
    assert result.run.id == RUN_ID and result.confidence == "high"


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
