"""Error taxonomy. One exception per behavior the caller must choose between.

- NimbleAuthError: the key is missing or rejected. Nothing retries this; it is raised to the developer.
- NimbleCreateAmbiguousError: the outcome of the create request is unknown: no response, a timeout, a 408 or a
  5xx. Nimble's billing table classifies all of these as "the request may have reached Nimble"; a run may be
  running and billed. Creation is billable and not idempotent with no idempotency key, so this is never
  resubmitted automatically; reconcile against the run history instead.
- NimbleRateLimitError: 429 after the retry budget (reads) or on creation, where Nimble says nothing was created,
  so a later attempt is safe.
- NimbleTimeoutError: the overall deadline passed. Carries the run identity once a run exists; the run keeps
  going on Nimble's side and its result stays fetchable with `NimbleClient.collect`.
- NimbleRunFailedError: Nimble finished the run with status "failed".
- NimbleAPIError: any other non-success response, with status code, raw body text, and the phase it came from.

Error bodies are kept as text, never parsed on the assumption that they are JSON: the 401 body is plain text.
The stored API key is never interpolated into a message by this package; a body echoed by the server is
surfaced as the server sent it, clipped.
"""

from __future__ import annotations


class NimbleError(Exception):
    """Base class for every error raised by this package."""


class NimbleAuthError(NimbleError):
    """The API key is missing, malformed, or rejected (401 / 403)."""


class NimbleCreateAmbiguousError(NimbleError):
    """The create request's outcome is unknown. A run may or may not exist; do not create another blindly."""

    def __init__(self, cause: str, *, status_code: int | None = None, body: str = "") -> None:
        self.cause = cause
        self.status_code = status_code
        self.body = body
        what = f"was answered with HTTP {status_code}" if status_code else "got no response"
        super().__init__(
            f"Nimble's run-creation request {what} ({_clip(cause, 120)}). Nimble classifies this outcome as "
            "unknown: a run may already be running and billed, so the request was not resubmitted. Check the run "
            "history in the Nimble dashboard before creating another."
        )


class NimbleAPIError(NimbleError):
    """A non-success HTTP response that is not an auth failure."""

    def __init__(
        self,
        status_code: int,
        body: str,
        *,
        method: str = "",
        path: str = "",
        phase: str = "",
        run_id: str | None = None,
        web_search_agent_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.method = method
        self.path = path
        self.phase = phase
        self.run_id = run_id
        self.web_search_agent_id = web_search_agent_id
        where = f" {method} {path}" if method else ""
        super().__init__(f"Nimble API returned {status_code}{where}: {_clip(body)}")


class NimbleRateLimitError(NimbleAPIError):
    """429: on creation a definite rejection; on reads, persisted past the retry budget."""


class NimbleRunFailedError(NimbleAPIError):
    """The run reached status 'failed'. `body` holds Nimble's error text."""

    def __init__(self, run_id: str, body: str, *, web_search_agent_id: str | None = None) -> None:
        NimbleError.__init__(self, f"Nimble run {run_id} failed: {_clip(body)}")
        self.status_code = 200
        self.body = body
        self.method = ""
        self.path = ""
        self.phase = "poll"
        self.run_id = run_id
        self.web_search_agent_id = web_search_agent_id


class NimbleTimeoutError(NimbleError):
    """The deadline passed. Once a run exists its identity is here, and the run is still going at Nimble."""

    def __init__(
        self,
        run_id: str | None,
        elapsed_s: float,
        deadline_s: float,
        *,
        web_search_agent_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.web_search_agent_id = web_search_agent_id
        self.elapsed_s = elapsed_s
        self.deadline_s = deadline_s
        which = f"run {run_id}" if run_id else "the request"
        tail = "; the run continues on Nimble's side and its result stays fetchable" if run_id else ""
        super().__init__(f"Nimble did not finish {which} within {deadline_s:.0f}s (waited {elapsed_s:.1f}s){tail}")


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
