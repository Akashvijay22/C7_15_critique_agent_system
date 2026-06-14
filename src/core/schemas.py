"""
Structured output contracts.

Every agent emits an ``AgentReport``. The coordinator merges those into a
single ``ConsolidatedReport``. Keeping these as strict Pydantic models gives us
three things at once:

1. A machine-checkable contract for the LLM's JSON output (validation + repair).
2. A stable shape for the Streamlit UI to render against.
3. Traceable citations: every Finding carries ``citations`` that point back to
   entries in the design knowledge base.

This module has no I/O and no third-party-service dependencies, so it can be
imported anywhere (agents, RAG, UI, tests) without side effects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """Ordered severity buckets. Order matters for prioritisation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3}[self.value]


class Effort(str, Enum):
    """Rough implementation cost, used to surface 'quick wins'."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Citation(BaseModel):
    """A reference back to a knowledge-base entry that grounds a finding."""

    kb_id: str = Field(..., description="ID of the knowledge-base entry, e.g. 'nn-h2'")
    title: str = Field(..., description="Human-readable name of the principle/law")
    source: str = Field(..., description="Where the principle comes from")
    url: Optional[str] = Field(default=None, description="Canonical reference URL")


class Region(BaseModel):
    """Normalized bounding box (0-1) locating a finding on a specific screen.

    Coordinates are fractions of image width/height so they survive resizing.
    ``image_index`` is the 0-based position of the uploaded image the box refers to.
    """

    image_index: int = Field(default=0, ge=0, description="Which uploaded image (0-based)")
    x: float = Field(..., ge=0.0, le=1.0, description="Left edge, fraction of width")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge, fraction of height")
    w: float = Field(..., ge=0.0, le=1.0, description="Width, fraction of image width")
    h: float = Field(..., ge=0.0, le=1.0, description="Height, fraction of image height")


class Finding(BaseModel):
    """A single, actionable critique produced by an agent."""

    title: str = Field(..., description="Short, specific headline for the issue")
    description: str = Field(..., description="What is wrong and why it matters to the user")
    severity: Severity = Field(default=Severity.MEDIUM)
    location: str = Field(
        default="general",
        description="Where on screen this applies, e.g. 'primary CTA', 'top nav'",
    )
    region: Optional[Region] = Field(
        default=None,
        description="Normalized bounding box of the issue on screen, if locatable",
    )
    recommendation: str = Field(..., description="Concrete, do-this-next fix")
    effort: Effort = Field(default=Effort.MEDIUM)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("title", "description", "recommendation")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @property
    def priority_score(self) -> float:
        """Lower is more urgent. Severity dominates, effort and confidence tune it."""
        effort_penalty = {"low": 0.0, "medium": 0.3, "high": 0.6}[self.effort.value]
        return self.severity.rank + effort_penalty - (self.confidence * 0.2)


class AgentReport(BaseModel):
    """One specialist agent's complete output for a single screen."""

    agent: str = Field(..., description="Agent name, e.g. 'Visual Analysis Agent'")
    summary: str = Field(..., description="2-3 sentence overall read")
    score: int = Field(default=70, ge=0, le=100, description="Health score for this lens")
    findings: list[Finding] = Field(default_factory=list)
    model: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ConsolidatedReport(BaseModel):
    """The coordinator's merged, de-duplicated, prioritised verdict."""

    overall_score: int = Field(default=70, ge=0, le=100)
    executive_summary: str = ""
    quick_wins: list[Finding] = Field(default_factory=list)
    prioritised_findings: list[Finding] = Field(default_factory=list)
    agent_reports: list[AgentReport] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_markdown(self) -> str:
        """Render the report as a portable Markdown brief (for export)."""
        lines: list[str] = []
        lines.append(f"# UI/UX Critique Report\n")
        lines.append(f"**Overall score:** {self.overall_score}/100  ")
        lines.append(f"**Generated:** {self.created_at}\n")
        lines.append(f"## Executive summary\n\n{self.executive_summary}\n")

        if self.quick_wins:
            lines.append("## Quick wins\n")
            for f in self.quick_wins:
                lines.append(f"- **{f.title}** — {f.recommendation}")
            lines.append("")

        lines.append("## Prioritised findings\n")
        for i, f in enumerate(self.prioritised_findings, 1):
            lines.append(f"### {i}. {f.title}  `[{f.severity.value}]`")
            lines.append(f"*Location:* {f.location} · *Effort:* {f.effort.value} "
                         f"· *Confidence:* {f.confidence:.0%}\n")
            lines.append(f"{f.description}\n")
            lines.append(f"**Recommendation:** {f.recommendation}\n")
            if f.citations:
                refs = "; ".join(
                    f"{c.title} ({c.source})" for c in f.citations
                )
                lines.append(f"*Grounded in:* {refs}\n")
        return "\n".join(lines)
