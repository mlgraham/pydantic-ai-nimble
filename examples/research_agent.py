"""A PydanticAI agent that researches a question on the live web through Nimble.

    uv run python examples/research_agent.py "What changed in the EU AI Act this month?"

Needs two keys in .env (or the environment): NIMBLE_API_KEY for Nimble, and a model provider key for the agent:
OPENAI_API_KEY, OPENROUTER_API_KEY or ANTHROPIC_API_KEY, checked in that order. Pick any other provider with
NIMBLE_EXAMPLE_MODEL set to a PydanticAI model string ("provider:model-name") plus that provider's key.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from pydantic_ai import Agent

from pydantic_ai_nimble import NimbleToolset

DEFAULT_QUESTION = "What did the European Commission publish about general-purpose AI obligations in September 2026?"

# The first provider whose key is set wins; override with NIMBLE_EXAMPLE_MODEL.
DEFAULT_MODELS = (
    ("OPENAI_API_KEY", "openai:gpt-5.4"),
    ("OPENROUTER_API_KEY", "openrouter:openai/gpt-5.4"),
    ("ANTHROPIC_API_KEY", "anthropic:claude-sonnet-5-5"),
)


def pick_model() -> str:
    if chosen := os.environ.get("NIMBLE_EXAMPLE_MODEL"):
        return chosen
    for variable, model in DEFAULT_MODELS:
        if os.environ.get(variable):
            return model
    raise SystemExit(
        "No model provider key found. Set OPENAI_API_KEY, OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env, "
        "or set NIMBLE_EXAMPLE_MODEL to a PydanticAI model string and its provider's key."
    )


def load_dotenv(path: Path) -> None:
    """Minimal .env loader so the example has no extra dependency. Existing variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'").strip('"'))


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("pydantic_ai_nimble").setLevel(logging.INFO)
    log = logging.getLogger("pydantic_ai_nimble")

    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    model = pick_model()
    log.info("model: %s", model)

    toolset = NimbleToolset()  # raises NimbleAuthError here if NIMBLE_API_KEY is missing: no model call is made
    agent = Agent(
        model,
        toolsets=[toolset],
        instructions=(
            "You answer questions about current events and documents. Use nimble_research for anything that "
            "needs live or verifiable information. Keep the [n] markers from the research in your answer and "
            "finish with the sources you used."
        ),
    )

    started = time.monotonic()
    result = agent.run_sync(question)
    elapsed = time.monotonic() - started

    print("\n" + result.output.strip() + "\n")
    research = toolset.last_result
    if research is None:
        print("(the model answered without calling nimble_research)")
    else:
        print(
            f"nimble run {research.run.id}: {research.confidence} confidence, {len(research.sources)} sources, "
            f"{len(research.claims)} graded claims, {research.elapsed_s:.1f}s in Nimble, {elapsed:.1f}s end to end"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
