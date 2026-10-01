"""The PydanticAI surface: one toolset, one tool.

    agent = Agent("anthropic:claude-sonnet-5-5", toolsets=[NimbleToolset()])

The toolset owns a `NimbleClient`. The tool runs a Web Search Agent task to completion under the client's
deadline and returns the compact text the model reads. Transient failures (timeout, rate limit, 5xx, transport,
a failed run) become one `ModelRetry`, so the model can narrow the query or try again. Auth failures and other
4xx responses propagate to the developer unchanged: the model cannot fix those.
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
        default_effort: effort used when the model does not choose one. "low" finishes in under a minute.
        max_retries: how many `ModelRetry` rounds the agent allows this tool. 1 means exactly one second try.
        max_sources / max_excerpt_chars: bounds on the compact text handed to the model.
        on_result: optional callback receiving every full `ResearchResult`; also appended to `self.results`.
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
        self.add_function(self._research, takes_ctx=False, name=tool_name)

    async def _research(self, query: str, effort: ToolEffort | None = None) -> str:
        """Research a question on the live web with Nimble's Web Search Agent and return a cited answer.

        Use this for anything that needs current or verifiable information: news, prices, regulations,
        documentation, people, companies. The answer contains [n] markers; the numbered sources that follow
        resolve them. Quote the markers when you use the information.

        Args:
            query: The research question in plain language, with any constraints such as dates, regions,
                or preferred sources. Ask one question per call.
            effort: How much research to do. "low" is fastest, about a minute. "medium" and "high" search
                deeper and take several minutes. Leave unset unless the first answer was too thin.
        """
        chosen = effort or self.default_effort
        try:
            result = await self.client.research(query, effort=chosen)
        except NimbleAuthError:
            raise
        except NimbleTimeoutError as error:
            raise ModelRetry(
                f"Nimble did not finish within {error.deadline_s:.0f}s. Ask a narrower question, or try again."
            ) from error
        except NimbleRateLimitError as error:
            raise ModelRetry("Nimble is rate limiting requests. Wait a moment and try again.") from error
        except NimbleRunFailedError as error:
            raise ModelRetry(
                f"Nimble's research run failed ({error.body}). Rephrase the question and try again."
            ) from error
        except NimbleAPIError as error:
            if error.status_code == 0 or error.status_code >= 500:
                raise ModelRetry(f"Nimble returned a transient error ({error.status_code}). Try again.") from error
            raise
        self.results.append(result)
        if self.on_result is not None:
            self.on_result(result)
        return result.compact(max_sources=self.max_sources, max_excerpt_chars=self.max_excerpt_chars)

    @property
    def last_result(self) -> ResearchResult | None:
        return self.results[-1] if self.results else None
