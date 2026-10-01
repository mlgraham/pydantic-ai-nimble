"""Async client for Nimble's Web Search Agent: create a run, poll it under one deadline, fetch the result.

Shape measured on 2026-09-30 (tests/fixtures):
    POST {base}/agents/runs                               -> 202 RunInfo (status queued, is_active true)
    GET  {base}/agents/{web_search_agent_id}/runs/{id}    -> 200 RunInfo; poll while is_active
    GET  {base}/agents/{web_search_agent_id}/runs/{id}/result -> 200 {run, output: {content, trust, type}}
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

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.deadline_s - self.elapsed()

    def timeout(self, run_id: str | None) -> NimbleTimeoutError:
        return NimbleTimeoutError(run_id, self.elapsed(), self.deadline_s)


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
        """Run one Web Search Agent task end to end under a single deadline."""
        budget = _Budget(deadline_s if deadline_s is not None else self.settings.deadline_s)

        body: dict[str, Any] = {"input": query, **extra}
        if effort:
            body["effort"] = effort
        if use_case:
            body["use_case"] = use_case
        run = await self.create_run(body, budget=budget)
        logger.info("nimble run %s created (effort=%s)", run.id, run.effort)

        delay = self.settings.poll_initial_s
        polls = 0
        while run.is_active:
            left = budget.remaining()
            if left <= 0:
                raise budget.timeout(run.id)
            await asyncio.sleep(min(delay, left))
            delay = min(delay * 2, self.settings.poll_max_s)
            if budget.remaining() <= 0:
                raise budget.timeout(run.id)
            run = await self.get_run(run.web_search_agent_id, run.id, budget=budget)
            polls += 1
            logger.debug("nimble run %s poll %d: status=%s", run.id, polls, run.status)

        if run.status != "completed":
            raise NimbleRunFailedError(run.id, run.error or run.status)
        result = await self.get_result(run.web_search_agent_id, run.id, budget=budget)
        elapsed = budget.elapsed()
        logger.info("nimble run %s completed in %.1fs after %d polls", run.id, elapsed, polls)
        return ResearchResult.from_payloads(run, result, elapsed)

    async def create_run(self, body: dict[str, Any], *, budget: _Budget | None = None) -> RunInfo:
        payload = await self._request("POST", "/agents/runs", json=body, budget=budget)
        return RunInfo.model_validate(payload)

    async def get_run(self, agent_id: str, run_id: str, *, budget: _Budget | None = None) -> RunInfo:
        payload = await self._request("GET", f"/agents/{agent_id}/runs/{run_id}", budget=budget, run_id=run_id)
        return RunInfo.model_validate(payload)

    async def get_result(self, agent_id: str, run_id: str, *, budget: _Budget | None = None) -> dict[str, Any]:
        path = f"/agents/{agent_id}/runs/{run_id}/result"
        payload = await self._request("GET", path, budget=budget, run_id=run_id)
        if not isinstance(payload, dict):
            raise NimbleAPIError(200, str(payload), method="GET", path=f"/agents/{agent_id}/runs/{run_id}/result")
        return payload

    # ---------------------------------------------------------------- transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        budget: _Budget | None = None,
        run_id: str | None = None,
    ) -> Any:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        headers = {**self.settings.auth_header(), "Accept": "application/json"}
        attempt = 0
        backoff = self.settings.retry_backoff_s
        while True:
            timeout = self.settings.request_timeout_s
            if budget is not None:
                left = budget.remaining()
                if left <= 0:
                    raise budget.timeout(run_id)
                timeout = min(timeout, left)
            try:
                response = await self._http.request(method, url, json=json, headers=headers, timeout=timeout)
            except httpx.TimeoutException as error:
                if budget is not None and budget.remaining() <= 0:
                    raise budget.timeout(run_id) from error
                if attempt >= self.settings.max_retries:
                    raise NimbleAPIError(0, f"transport timeout: {error}", method=method, path=path) from error
            except httpx.TransportError as error:
                if attempt >= self.settings.max_retries:
                    raise NimbleAPIError(0, f"transport error: {error}", method=method, path=path) from error
            else:
                status = response.status_code
                if status in (401, 403):
                    raise NimbleAuthError(f"Nimble rejected the API key ({status}): {_text(response)}")
                if status < 300:
                    return _json_or_text(response)
                if status not in RETRYABLE_STATUSES or attempt >= self.settings.max_retries:
                    if status == 429:
                        raise NimbleRateLimitError(status, _text(response), method=method, path=path)
                    raise NimbleAPIError(status, _text(response), method=method, path=path)
                retries = self.settings.max_retries
                logger.warning("nimble %s %s returned %d, retry %d/%d", method, path, status, attempt + 1, retries)
            attempt += 1
            wait = backoff
            if budget is not None:
                left = budget.remaining()
                if left <= 0:
                    raise budget.timeout(run_id)
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
