"""Typed views of Nimble's Web Search Agent responses, shaped by the fixtures captured from a live run.

Run objects (create, poll) carry `id`, `web_search_agent_id`, `status`, `is_active`. The result object carries
`output.content` (markdown) and `output.trust` with a run-level confidence, a flat `sources` list, and graded
`claims` whose citations may or may not include verbatim excerpts.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["high", "medium", "low", "pre_existing"]
Effort = Literal["low", "medium", "high", "x-high"]
UseCase = Literal["research", "enrichment", "dataset_building"]


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class RunInfo(_Lenient):
    """The run object returned by create and by every poll."""

    id: str
    web_search_agent_id: str
    status: str
    is_active: bool
    effort: str | None = None
    prompt: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class Source(_Lenient):
    """One entry of `output.trust.sources`."""

    url: str
    title: str | None = None
    type: str | None = None  # "primary" | "secondary"
    source_category: str | None = None  # "official" | "news" | "social" | "academic"


class Citation(_Lenient):
    """One citation under a claim. `excerpts` was null on a low-effort run; treat it as optional."""

    url: str
    title: str | None = None
    excerpts: list[str] | None = None
    source_type: str | None = None
    source_category: str | None = None


class Claim(_Lenient):
    callout: int | None = None
    confidence: str | None = None
    reasoning: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class Trust(_Lenient):
    confidence: str | None = None
    reasoning: str | None = None
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


class ResearchResult(_Lenient):
    """What `NimbleClient.research` returns. Kept whole; `compact()` is the view the model sees."""

    run: RunInfo
    answer: str
    confidence: str | None
    reasoning: str | None
    sources: list[Source]
    claims: list[Claim]
    trust: dict[str, Any]
    elapsed_s: float

    @classmethod
    def from_payloads(cls, run: RunInfo, result: dict[str, Any], elapsed_s: float) -> ResearchResult:
        output = result.get("output") or {}
        trust_raw = output.get("trust") or {}
        trust = Trust.model_validate(trust_raw)
        return cls(
            run=run,
            answer=str(output.get("content") or ""),
            confidence=trust.confidence,
            reasoning=trust.reasoning,
            sources=trust.sources,
            claims=trust.claims,
            trust=trust_raw,
            elapsed_s=elapsed_s,
        )

    def cited_markers(self) -> list[int]:
        """The distinct `[n]` markers that appear in the answer, ascending."""
        return sorted({int(match) for match in re.findall(r"\[(\d+)\]", self.answer)})

    def compact(self, max_sources: int = 12, max_excerpt_chars: int = 200) -> str:
        """The string handed to the model.

        Nimble's answer carries `[n]` markers that index `trust.sources` one-based (measured: nine markers,
        nine sources, in order). The list below keeps those numbers, so the model can quote `[3]` and have it
        resolve. A claim whose `callout` matches a marker adds its confidence grade; an excerpt line appears only
        when a citation carries non-empty excerpts. Sources beyond `max_sources` are dropped unless cited.
        """
        lines = [f"Answer (confidence: {self.confidence or 'unknown'}, {self.elapsed_s:.1f}s):", self.answer.strip()]
        if not self.sources:
            return "\n".join(lines)
        cited = set(self.cited_markers())
        grade_by_marker = {claim.callout: claim.confidence for claim in self.claims if claim.callout is not None}
        excerpt_by_url: dict[str, str] = {}
        for claim in self.claims:
            for citation in claim.citations:
                if citation.excerpts and citation.url not in excerpt_by_url:
                    excerpt_by_url[citation.url] = citation.excerpts[0]
        lines += ["", "Sources (numbers match the [n] markers above):"]
        for index, source in enumerate(self.sources, start=1):
            if index > max_sources and index not in cited:
                continue
            tags = ", ".join(tag for tag in (source.source_category, source.type) if tag)
            grade = grade_by_marker.get(index)
            head = f"[{index}] {source.title}" if source.title else f"[{index}]"
            meta = "; ".join(part for part in (tags, f"confidence: {grade}" if grade else "") if part)
            lines.append(f"{head} ({meta})" if meta else head)
            lines.append(f"    {source.url}")
            if excerpt := excerpt_by_url.get(source.url):
                lines.append(f'    "{_clip_excerpt(excerpt, max_excerpt_chars)}"')
        return "\n".join(lines)


def _clip_excerpt(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
