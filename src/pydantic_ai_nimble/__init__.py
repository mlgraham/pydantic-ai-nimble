"""Nimble's Web Search Agent for PydanticAI.

from pydantic_ai import Agent
from pydantic_ai_nimble import NimbleToolset

agent = Agent("openai:gpt-5.4", toolsets=[NimbleToolset()])
"""

from pydantic_ai_nimble.client import NimbleClient
from pydantic_ai_nimble.errors import (
    NimbleAPIError,
    NimbleAuthError,
    NimbleError,
    NimbleRateLimitError,
    NimbleRunFailedError,
    NimbleTimeoutError,
)
from pydantic_ai_nimble.models import Citation, Claim, ResearchResult, RunInfo, Source
from pydantic_ai_nimble.settings import NimbleSettings
from pydantic_ai_nimble.toolset import NimbleToolset

__all__ = [
    "Citation",
    "Claim",
    "NimbleAPIError",
    "NimbleAuthError",
    "NimbleClient",
    "NimbleError",
    "NimbleRateLimitError",
    "NimbleRunFailedError",
    "NimbleSettings",
    "NimbleTimeoutError",
    "NimbleToolset",
    "ResearchResult",
    "RunInfo",
    "Source",
]
__version__ = "0.1.1"
