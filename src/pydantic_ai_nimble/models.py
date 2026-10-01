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

    def answer_index(self) -> dict[int, str]:
        """The numbered source list Nimble writes at the end of its own answer, as marker -> url.

        Empty when the answer carries no such list. This is the authoritative resolution of the answer's `[n]`
        markers; the flat `trust.sources` inventory is not in the same order (measured on a captured run).
        """
        found: dict[int, str] = {}
        for marker, url in re.findall(r"^\s*\[(\d+)\][^\n]*?(https?://\S+)", self.answer, flags=re.MULTILINE):
            found.setdefault(int(marker), url.rstrip(".,;)"))
        return found

    def graded_claims(self) -> list[Claim]:
        """Claims that carry a callout and at least one citation, in callout order."""
        return sorted(
            (claim for claim in self.claims if claim.callout is not None and claim.citations),
            key=lambda claim: claim.callout or 0,
        )

    def compact(self, max_excerpt_chars: int = 200, urls_per_claim: int = 1, max_sources: int = 12) -> str:
        """The string handed to the model.

        Nimble's answer is kept whole, including the numbered source index it ends with; nothing is renumbered.
        Underneath, the trust report is attached as one line per graded claim, keyed by the claim's `callout`,
        which is the `[n]` marker the answer uses for it: Nimble's confidence grade, the cited page, an excerpt
        when Nimble returned one. Markers without a graded claim are not listed. If a claim cites a different
        page than the answer's own index gives for that marker, the line says so instead of hiding it. A result
        with no claims gets a capped inventory of the pages Nimble consulted, labelled as not being the answer's
        numbering.
        """
        header = f"Answer (Nimble confidence: {self.confidence or 'unknown'}, {self.elapsed_s:.1f}s):"
        lines = [header, self.answer.strip()]
        claims = self.graded_claims()
        index = self.answer_index()
        if claims:
            lines += ["", "Nimble's confidence per cited claim (numbers are the [n] markers in the answer):"]
            for claim in claims:
                marker = claim.callout
                lines.append(f"[{marker}] {claim.confidence or 'ungraded'}")
                for citation in claim.citations[:urls_per_claim]:
                    title = f"{citation.title}: " if citation.title else ""
                    lines.append(f"    {title}{citation.url}")
                    if citation.excerpts:
                        lines.append(f'    "{_clip_excerpt(citation.excerpts[0], max_excerpt_chars)}"')
                extra = len(claim.citations) - urls_per_claim
                if extra > 0:
                    lines.append(f"    (+{extra} more citation{'s' if extra > 1 else ''})")
                expected = index.get(marker) if marker is not None else None
                if expected and expected not in {citation.url.rstrip(".,;)") for citation in claim.citations}:
                    lines.append(f"    (note: the answer's own [{marker}] entry is a different page: {expected})")
        elif self.sources:
            lines += ["", f"Pages Nimble consulted (an inventory, not the answer's numbering; first {max_sources}):"]
            lines += [f"    {source.url}" for source in self.sources[:max_sources]]
        return "\n".join(lines)


def _clip_excerpt(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
