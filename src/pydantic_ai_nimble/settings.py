"""Configuration. The key comes from an explicit argument or NIMBLE_API_KEY; it is stored as a SecretStr so the
stored value never appears in repr or logs, and this package never interpolates it into a message. Text a server
echoes back is surfaced as sent."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from pydantic_ai_nimble.errors import NimbleAuthError

ENV_API_KEY = "NIMBLE_API_KEY"
ENV_BASE_URL = "NIMBLE_BASE_URL"
ENV_DEADLINE = "NIMBLE_DEADLINE_S"
DEFAULT_BASE_URL = "https://sdk.nimbleway.com/v2"

MISSING_KEY_MESSAGE = (
    f"{ENV_API_KEY} is not set. Get a key at https://app.nimbleway.com and put it in .env (see .env.example), "
    "or pass api_key= explicitly."
)


class NimbleSettings(BaseModel):
    """Everything the client needs. Construct directly or with `NimbleSettings.from_env()`."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    base_url: str = DEFAULT_BASE_URL
    deadline_s: float = Field(
        default=300.0,
        gt=0,
        description="Wall-clock budget for one research call. Low effort finishes in about a minute; medium has "
        "measured over 90 s; Nimble suggests much longer for high.",
    )
    poll_initial_s: float = Field(default=1.0, gt=0)
    poll_max_s: float = Field(default=8.0, gt=0)
    request_timeout_s: float = Field(default=30.0, gt=0, description="Per-HTTP-request timeout.")
    max_retries: int = Field(default=3, ge=0, description="Retries for 429, 5xx and transport errors.")
    retry_backoff_s: float = Field(default=0.5, ge=0, description="First retry delay; doubles each time.")

    @field_validator("api_key")
    @classmethod
    def _key_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError(MISSING_KEY_MESSAGE)
        return value

    @classmethod
    def from_env(cls, api_key: str | None = None, **overrides: object) -> NimbleSettings:
        """Build settings from the environment. An explicit `api_key` wins over the variable."""
        key = api_key or os.environ.get(ENV_API_KEY, "").strip()
        if not key:
            raise NimbleAuthError(MISSING_KEY_MESSAGE)
        values: dict[str, object] = {"api_key": key}
        if base_url := os.environ.get(ENV_BASE_URL, "").strip():
            values["base_url"] = base_url
        if deadline := os.environ.get(ENV_DEADLINE, "").strip():
            values["deadline_s"] = float(deadline)
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
