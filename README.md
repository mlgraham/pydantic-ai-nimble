# pydantic-ai-nimble

Nimble's Web Search Agent as a [PydanticAI](https://ai.pydantic.dev) toolset. One import, one line in the
`Agent` constructor, and the agent can research the live web and come back with a cited, confidence-graded answer.

```python
from pydantic_ai import Agent
from pydantic_ai_nimble import NimbleToolset

agent = Agent("openai:gpt-5.4", toolsets=[NimbleToolset()])  # any PydanticAI model works
result = agent.run_sync("What changed in the EU AI Act this month?")
```

## Quick start

```sh
git clone <this repo> && cd pydantic-ai-nimble
uv sync                                   # Python 3.11+, installs pydantic-ai and httpx
cp .env.example .env                      # put NIMBLE_API_KEY and one model provider key in it
uv run python examples/research_agent.py "What changed in the EU AI Act this month?"
uv run pytest -q                          # 24 tests, no network, under two seconds
```

Get a Nimble key at [app.nimbleway.com](https://app.nimbleway.com). The example uses the first provider key it
finds: `OPENAI_API_KEY` (GPT-5.4), then `OPENROUTER_API_KEY` (GPT-5.4 via OpenRouter), then `ANTHROPIC_API_KEY`
(Claude Sonnet 5.5). Set `NIMBLE_EXAMPLE_MODEL` to any PydanticAI model string to choose a different model or
provider. The model and the research are independent: the toolset works with whatever model PydanticAI runs.

## What it does

`NimbleToolset` registers one tool, `nimble_research(query, effort)`. The tool's schema comes from its docstring,
so the model knows when to call it and what to put in `query`. When called, the toolset's `NimbleClient`:

1. `POST /v2/agents/runs` creates a Web Search Agent run (202, `is_active: true`).
2. Polls `GET /v2/agents/{agent_id}/runs/{id}` with backoff (1, 2, 4, 8 s, capped) until `is_active` is false,
   all under one deadline (120 s by default).
3. `GET .../result` fetches the answer (`output.content`) and the trust report (`output.trust`).
4. Parses it into a typed `ResearchResult` and hands the model a compact view: the answer, then the numbered
   sources that its `[n]` markers point at, with the confidence grade Nimble gave each claim.

The full `ResearchResult` (every source, every claim, the raw trust report) stays on the toolset as
`toolset.last_result` / `toolset.results`, or flows to an `on_result` callback, so your code keeps everything
the model did not need to read.

Failures map to one behavior each. A missing or rejected key raises `NimbleAuthError` before any model call
is made. Timeouts, rate limits, 5xx and transport errors become a single `ModelRetry`, so the model can narrow
the question or try again once. Other 4xx responses propagate to you unchanged.

## Why this shape

**PydanticAI**, because its typed tools and `FunctionToolset` map directly onto Nimble's typed request and
graded response, and because a toolset is the unit PydanticAI composes: `toolsets=[NimbleToolset()]` is the
whole install, and the same object can be filtered, prefixed, or combined with the framework's own wrappers.

**The Agent API directly over thin `httpx`**, not the MCP server and not the `nimble_python` SDK. Web Search
Agent runs are asynchronous, so the interesting engineering is the lifecycle: one deadline across create, poll
and fetch; backoff; a run id on every timeout; error bodies treated as text (the 401 body is not JSON). Owning
that loop is what makes the integration something a maintainer can reason about, and `httpx` keeps the
dependency footprint to what PydanticAI already pulls in. Everything is mockable with `respx`, and the test
fixtures are real responses captured from a live run.

**One tool, with the client doing the polling.** Nimble's own [proposal for pydantic-ai](https://github.com/pydantic/pydantic-ai/issues/9211)
and its [harness PR](https://github.com/pydantic/pydantic-ai-harness/pull/476) expose `agent_run_start`,
`agent_run_status` and `agent_run_result` as separate tools and leave the polling to the model. That is flexible,
but it spends model turns and tokens on waiting, and it lets the model forget to fetch the result. This package
makes the opposite bet: the model asks one question and gets one answer; the wait is the client's problem.
Nimble also ships [langchain-nimble](https://github.com/Nimbleway/langchain-nimble), a LlamaIndex tool spec and a
Mastra package; none of them targets PydanticAI's toolset interface, which is the gap this fills.

## Tradeoffs and limits

- **Compact output by design.** The model sees the answer plus the cited sources, bounded in size. Excerpts are
  included only when Nimble returns them (they were `null` on the captured low-effort run), so the model is never
  shown a quote that does not exist.
- **Numbering follows Nimble.** The `[n]` markers in Nimble's answer index its `trust.sources` list one-based;
  the compact view keeps those numbers rather than renumbering, so citations stay resolvable.
- **One tool.** Only the Web Search Agent. Nimble's search, extract, map and crawl endpoints are out of scope;
  the toolset pattern makes them additive.
- **No streaming.** The run's progress events are not surfaced; the deadline is the only feedback during a run.
- **Results live on the toolset instance.** `toolset.results` is per instance, not per agent run. Use
  `on_result` or one toolset per run if you need isolation under concurrency.
- **Effort is capped at `high` in the tool schema.** Nimble's `x-high` is reachable through `NimbleClient`
  directly; it was left out of the model-facing enum because it can take many minutes.
- **The sources are the guarantee, not the conclusion.** On one question, two models given the same Nimble result
  reached different conclusions about what it meant. The toolset delivers the same cited sources either way;
  interpretation is the model's. Keep the `[n]` markers in the agent's instructions so readers can check.
- **Pinned to the measured shape.** Models are lenient (`extra="ignore"`), so new fields will not break
  parsing, but a renamed field would. The fixtures in `tests/fixtures` record exactly what was observed.

## Next

- An MCP transport option (`MCPServerStreamableHTTP` against `mcp.nimbleway.com`) for users who already run
  the Nimble MCP server, sharing the same compact view.
- Streaming run events into PydanticAI's event stream so long runs show progress.
- A `NimbleDeps` object so results attach to the agent run rather than the toolset.
- Offer this client to the pydantic-ai maintainers as the implementation under a `NimbleAgent` capability, and
  port the same client/tool shape to Mastra and the Cloudflare Agents SDK with shared fixtures.

## Where AI tools were used

Claude Code was used throughout, under a dated project ledger. It drafted the client, models, toolset, tests and
this README, and it ran the live capture that produced the fixtures. The design decisions were made and reviewed
by hand: the choice of PydanticAI and the direct API, the error taxonomy and which errors retry, the decision to
own the polling loop rather than expose it as tools, and the reading of the trust report that keeps Nimble's
citation numbering. Every number in this README was measured, not estimated.
