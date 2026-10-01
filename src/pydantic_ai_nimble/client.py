"""Async client for Nimble's Web Search Agent: create a run once, poll it under one deadline, fetch the result.

Shape measured on a live run (tests/fixtures):
    POST {base}/agents/runs                                   -> 202 RunInfo (status queued, is_active true)
    GET  {base}/agents/{web_search_agent_id}/runs/{id}        -> 200 RunInfo; poll while is_active
    GET  {base}/agents/{web_search_agent_id}/runs/{id}/result -> 200 {run, output: {content, trust, type}}

Rules, from Nimble's own connector documentation:
- Run creation is billable and not idempotent and there is no idempotency key, so the create call is issued
  exactly once. Per Nimble's billing table, only a 429 means nothing was created; no response, a timeout, a 408
  or a 5xx is unknown and raises `NimbleCreateAmbiguousError`, never resubmitted. Other 4xx are reported as is.
- A client-side deadline does not stop the run. A timeout carries the run identity and `collect()` fetches
  the result later.
- Reads (poll, result) are safe to retry and are, within the deadline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Any

import httpx

from pydantic_ai_nimble.errors import (
    NimbleAPIError,
    NimbleAuthError,
    NimbleCreateAmbiguousError,
    NimbleRateLimitError,
    NimbleRunFailedError,
    NimbleTimeoutError,
)
from pydantic_ai_nimble.models import Effort, ResearchResult, RunInfo, UseCase
from pydantic_ai_nimble.settings import NimbleSettings

logger = logging.getLogger("pydantic_ai_nimble")

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class _Budget:
    """One wall-clock budget for a whole research call: create, every poll, every retry sleep, the result."""

    def __init__(self, deadline_s: float) -> None:
        self.deadline_s = deadline_s
        self.started = time.monotonic()
        self.run_id: str | None = None
        self.web_search_agent_id: str | None = None
        self.create_in_flight = False

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.deadline_s - self.elapsed()

    def timeout(self) -> NimbleTimeoutError:
        return NimbleTimeoutError(
            self.run_id, self.elapsed(), self.deadline_s, web_search_agent_id=self.web_search_agent_id
        )


class NimbleClient:
    """One instance per application. Safe to share across agent runs.

    Pass an `httpx.AsyncClient` to control transport (tests pass a mocked one); otherwise one is created and
    closed by `aclose()` or the async context manager.
    """

    def __init__(
        self,
        settings: NimbleSettings | None = None,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or NimbleSettings.from_env(api_key=api_key)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient()

    def __repr__(self) -> str:
        return f"NimbleClient(base_url={self.settings.base_url!r}, deadline_s={self.settings.deadline_s})"

    async def __aenter__(self) -> NimbleClient:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # ---------------------------------------------------------------- public

    async def research(
        self,
        query: str,
        *,
        effort: Effort | None = None,
        use_case: UseCase | None = "research",
        deadline_s: float | None = None,
        **extra: Any,
    ) -> ResearchResult:
        """Create one run and see it through to a result, all inside a single wall-clock deadline."""
        body: dict[str, Any] = {"input": query, **extra}
        if effort:
            body["effort"] = effort
        if use_case:
            body["use_case"] = use_case
        budget = _Budget(deadline_s if deadline_s is not None else self.settings.deadline_s)
        try:
            async with asyncio.timeout(budget.deadline_s):
                run = await self.create_run(body, budget=budget)
                budget.run_id, budget.web_search_agent_id = run.id, run.web_search_agent_id
                logger.info("nimble run %s created (effort=%s)", run.id, run.effort)
                return await self._finish(run, budget)
        except TimeoutError as error:
            if budget.create_in_flight:
                # The deadline expired with the create request sent and unanswered: the same unknown as a timeout.
                raise NimbleCreateAmbiguousError(
                    f"deadline of {budget.deadline_s:.0f}s expired while the create request was in flight"
                ) from error
            raise budget.timeout() from error

    async def collect(
        self, web_search_agent_id: str, run_id: str, *, deadline_s: float | None = None
    ) -> ResearchResult:
        """Resume an existing run: poll it to completion and fetch its result. Creates nothing."""
        budget = _Budget(deadline_s if deadline_s is not None else self.settings.deadline_s)
        budget.run_id, budget.web_search_agent_id = run_id, web_search_agent_id
        try:
            async with asyncio.timeout(budget.deadline_s):
                run = await self.get_run(web_search_agent_id, run_id, budget=budget)
                return await self._finish(run, budget)
        except TimeoutError as error:
            raise budget.timeout() from error

    async def create_run(self, body: dict[str, Any], *, budget: _Budget | None = None) -> RunInfo:
        """Issue the create request exactly once. Anything but a clear answer is ambiguous and never re-sent."""
        payload = await self._request("POST", "/agents/runs", json=body, budget=budget, phase="create")
        return RunInfo.model_validate(payload)

    async def get_run(self, agent_id: str, run_id: str, *, budget: _Budget | None = None) -> RunInfo:
        payload = await self._request("GET", f"/agents/{agent_id}/runs/{run_id}", budget=budget, phase="poll")
        return RunInfo.model_validate(payload)

    async def get_result(self, agent_id: str, run_id: str, *, budget: _Budget | None = None) -> dict[str, Any]:
        path = f"/agents/{agent_id}/runs/{run_id}/result"
        payload = await self._request("GET", path, budget=budget, phase="result")
        if not isinstance(payload, dict):
            raise NimbleAPIError(
                200, str(payload), method="GET", path=path, phase="result", run_id=run_id, web_search_agent_id=agent_id
            )
        return payload

    # ---------------------------------------------------------------- lifecycle

    async def _finish(self, run: RunInfo, budget: _Budget) -> ResearchResult:
        delay = self.settings.poll_initial_s
        polls = 0
        while run.is_active:
            left = budget.remaining()
            if left <= 0:
                raise budget.timeout()
            await asyncio.sleep(min(delay, left))
            delay = min(delay * 2, self.settings.poll_max_s)
            run = await self.get_run(run.web_search_agent_id, run.id, budget=budget)
            polls += 1
            logger.debug("nimble run %s poll %d: status=%s", run.id, polls, run.status)
        if run.status != "completed":
            raise NimbleRunFailedError(run.id, run.error or run.status, web_search_agent_id=run.web_search_agent_id)
        result = await self.get_result(run.web_search_agent_id, run.id, budget=budget)
        if budget.remaining() < 0:
            raise budget.timeout()
        elapsed = budget.elapsed()
        logger.info("nimble run %s completed in %.1fs after %d polls", run.id, elapsed, polls)
        return ResearchResult.from_payloads(run, result, elapsed)

    # ---------------------------------------------------------------- transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        budget: _Budget | None = None,
        phase: str = "",
    ) -> Any:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        headers = {**self.settings.auth_header(), "Accept": "application/json"}
        creating = phase == "create"
        retries_allowed = 0 if creating else self.settings.max_retries
        ids: dict[str, Any] = {}
        if budget is not None:
            ids = {"run_id": budget.run_id, "web_search_agent_id": budget.web_search_agent_id}
        attempt = 0
        backoff = self.settings.retry_backoff_s
        while True:
            timeout = self.settings.request_timeout_s
            if budget is not None:
                left = budget.remaining()
                if left <= 0:
                    raise budget.timeout()
                timeout = min(timeout, left)
            if creating and budget is not None:
                budget.create_in_flight = True
            try:
                response = await self._http.request(method, url, json=json, headers=headers, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as error:
                if creating:
                    raise NimbleCreateAmbiguousError(f"{type(error).__name__}: {error}") from error
                if budget is not None and budget.remaining() <= 0:
                    raise budget.timeout() from error
                if attempt >= retries_allowed:
                    raise NimbleAPIError(
                        0, f"{type(error).__name__}: {error}", method=method, path=path, phase=phase, **ids
                    ) from error
            else:
                status = response.status_code
                if creating and budget is not None:
                    budget.create_in_flight = False
                if creating and (status == 408 or status >= 500):
                    raise NimbleCreateAmbiguousError(_text(response), status_code=status, body=_text(response))
                if status in (401, 403):
                    raise NimbleAuthError(f"Nimble rejected the API key ({status}): {_text(response)}")
                if status < 300:
                    return _json_or_text(response)
                if status not in RETRYABLE_STATUSES or attempt >= retries_allowed:
                    cls = NimbleRateLimitError if status == 429 else NimbleAPIError
                    raise cls(status, _text(response), method=method, path=path, phase=phase, **ids)
                logger.warning(
                    "nimble %s %s returned %d, retry %d/%d", method, path, status, attempt + 1, retries_allowed
                )
            attempt += 1
            wait = backoff
            if budget is not None:
                left = budget.remaining()
                if left <= 0:
                    raise budget.timeout()
                wait = min(backoff, left)
            await asyncio.sleep(wait)
            backoff *= 2


def _text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:  # pragma: no cover - defensive
        return "<unreadable body>"


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text
