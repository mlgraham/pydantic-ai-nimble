"""The PydanticAI surface: one toolset, one tool.

    agent = Agent("openai:gpt-5.4", toolsets=[NimbleToolset()])

The toolset owns a `NimbleClient`. The tool runs a Web Search Agent task to completion under the client's
deadline and returns the compact text the model reads.

What a retry means here, given that Nimble bills every run and offers no idempotency key:
- A run that timed out, or whose poll or result fetch failed after the client's own retries, is remembered by
  query. The model gets one `ModelRetry`; calling the tool again with the same query collects that same run
  instead of creating another. A different query is new work the model asked for.
- A create request that got no response (`NimbleCreateAmbiguousError`) and a create that was answered with an
  error are never turned into a `ModelRetry`: a run may already exist, or the request was definitely rejected
  for a reason the model cannot fix. They reach the developer. The one exception is a 429 on creation, which
  is a definite rejection and safe to try again.
- Auth failures and other 4xx responses reach the developer unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_nimble.client import NimbleClient
from pydantic_ai_nimble.errors import (
    NimbleAPIError,
    NimbleAuthError,
    NimbleCreateAmbiguousError,
    NimbleRateLimitError,
    NimbleRunFailedError,
    NimbleTimeoutError,
)
from pydantic_ai_nimble.models import ResearchResult
from pydantic_ai_nimble.settings import NimbleSettings

ToolEffort = Literal["low", "medium", "high"]
DEFAULT_TOOL_NAME = "nimble_research"


class NimbleToolset(FunctionToolset[Any]):
    """Exposes `nimble_research(query, effort)` to a PydanticAI agent.

    Parameters:
        client: a configured `NimbleClient`; built from `settings` / `api_key` / `NIMBLE_API_KEY` when omitted.
            A missing key raises `NimbleAuthError` here, before any model call.
        default_effort: effort used when the model does not choose one. "low" finishes in about a minute.
        max_retries: how many `ModelRetry` rounds the agent allows this tool. 1 means exactly one second try.
        max_sources / max_excerpt_chars: bounds on the compact text handed to the model.
        on_result: optional callback receiving every full `ResearchResult`; also appended to `self.results`.

    `self.pending` maps a query to the (web_search_agent_id, run_id) of a run that is still owed a result, so
    a developer can collect it later with `client.collect(...)` even after the agent gave up.
    """

    def __init__(
        self,
        client: NimbleClient | None = None,
        *,
        settings: NimbleSettings | None = None,
        api_key: str | None = None,
        default_effort: ToolEffort = "low",
        max_retries: int = 1,
        max_sources: int = 12,
        max_excerpt_chars: int = 200,
        tool_name: str = DEFAULT_TOOL_NAME,
        on_result: Callable[[ResearchResult], None] | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries, id="nimble")
        self.client = client or NimbleClient(settings, api_key=api_key)
        self.default_effort: ToolEffort = default_effort
        self.max_sources = max_sources
        self.max_excerpt_chars = max_excerpt_chars
        self.on_result = on_result
        self.results: list[ResearchResult] = []
        self.pending: dict[str, tuple[str, str]] = {}
        self.add_function(self._research, takes_ctx=False, name=tool_name)

    async def _research(self, query: str, effort: ToolEffort | None = None) -> str:
        """Research a question on the live web with Nimble's Web Search Agent and return a cited answer.

        Use this for anything that needs current or verifiable information: news, prices, regulations,
        documentation, people, companies. The answer contains [n] markers and ends with the numbered sources
        they refer to; a confidence grade per claim follows. Quote the markers when you use the information.

        Args:
            query: The research question in plain language, with any constraints such as dates, regions,
                or preferred sources. Ask one question per call.
            effort: How much research to do. "low" is fastest, about a minute. "medium" and "high" search
                deeper and take several minutes. Leave unset unless the first answer was too thin.
        """
        key = " ".join(query.split()).lower()
        chosen = effort or self.default_effort
        try:
            if key in self.pending:
                agent_id, run_id = self.pending[key]
                result = await self.client.collect(agent_id, run_id)
            else:
                result = await self.client.research(query, effort=chosen)
        except NimbleAuthError:
            raise
        except NimbleCreateAmbiguousError:
            raise
        except NimbleTimeoutError as error:
            if error.run_id and error.web_search_agent_id:
                self.pending[key] = (error.web_search_agent_id, error.run_id)
                raise ModelRetry(
                    f"Nimble is still researching this (run {error.run_id}); it did not finish within "
                    f"{error.deadline_s:.0f}s. Call nimble_research again with the same query to collect the "
                    "answer, or ask something narrower."
                ) from error
            raise
        except NimbleRunFailedError as error:
            self.pending.pop(key, None)
            raise ModelRetry(
                f"Nimble's research run failed ({error.body}). Rephrase the question and try again."
            ) from error
        except NimbleRateLimitError as error:
            if error.phase == "create":
                raise ModelRetry("Nimble is rate limiting new research runs. Wait a moment and try again.") from error
            self._remember(key, error)
            raise ModelRetry(
                "Nimble is rate limiting reads; the run continues. Call nimble_research again with the same query."
            ) from error
        except NimbleAPIError as error:
            if error.phase == "create":
                raise
            if error.run_id and error.web_search_agent_id and (error.status_code == 0 or error.status_code >= 500):
                self._remember(key, error)
                raise ModelRetry(
                    f"Nimble returned a transient error ({error.status_code}) while fetching run {error.run_id}; "
                    "the run continues. Call nimble_research again with the same query to collect it."
                ) from error
            raise
        self.pending.pop(key, None)
        self.results.append(result)
        if self.on_result is not None:
            self.on_result(result)
        return result.compact(self.max_sources, self.max_excerpt_chars)

    def _remember(self, key: str, error: NimbleAPIError) -> None:
        if error.run_id and error.web_search_agent_id:
            self.pending[key] = (error.web_search_agent_id, error.run_id)

    @property
    def last_result(self) -> ResearchResult | None:
        return self.results[-1] if self.results else None
