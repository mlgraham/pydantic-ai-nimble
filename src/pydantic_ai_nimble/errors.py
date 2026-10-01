"""Error taxonomy. One exception per behavior the caller must choose between.

- NimbleAuthError: the key is missing or rejected. Nothing retries this; it is raised to the developer.
- NimbleRateLimitError: 429 after the retry budget. Transient from the model's point of view.
- NimbleTimeoutError: the overall deadline passed while the run was still active. Carries the run id.
- NimbleRunFailedError: Nimble finished the run with status "failed".
- NimbleAPIError: any other non-success response, with status code and the raw body text.

Error bodies are kept as text, never parsed on the assumption that they are JSON: the 401 body is plain text.
No exception message ever contains the API key.
"""

from __future__ import annotations


class NimbleError(Exception):
    """Base class for every error raised by this package."""


class NimbleAuthError(NimbleError):
    """The API key is missing, malformed, or rejected (401 / 403)."""


class NimbleAPIError(NimbleError):
    """A non-success HTTP response that is not an auth failure."""

    def __init__(self, status_code: int, body: str, *, method: str = "", path: str = "") -> None:
        self.status_code = status_code
        self.body = body
        self.method = method
        self.path = path
        where = f" {method} {path}" if method else ""
        super().__init__(f"Nimble API returned {status_code}{where}: {_clip(body)}")


class NimbleRateLimitError(NimbleAPIError):
    """429 that persisted past the retry budget."""


class NimbleRunFailedError(NimbleAPIError):
    """The run reached status 'failed'. `body` holds Nimble's error text."""

    def __init__(self, run_id: str, body: str) -> None:
        self.run_id = run_id
        NimbleError.__init__(self, f"Nimble run {run_id} failed: {_clip(body)}")
        self.status_code = 200
        self.body = body
        self.method = ""
        self.path = ""


class NimbleTimeoutError(NimbleError):
    """The run was still active when the deadline passed."""

    def __init__(self, run_id: str | None, elapsed_s: float, deadline_s: float) -> None:
        self.run_id = run_id
        self.elapsed_s = elapsed_s
        self.deadline_s = deadline_s
        which = f"run {run_id}" if run_id else "the request"
        super().__init__(f"Nimble did not finish {which} within {deadline_s:.0f}s (waited {elapsed_s:.1f}s)")


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
