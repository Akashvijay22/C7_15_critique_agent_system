"""
Consolidation logic and prompts.

Pure functions (no LangGraph, no Streamlit) so they are trivially testable:
- semantic de-duplication of findings via embedding cosine similarity,
- severity-weighted scoring,
- the executive-summary prompt,
- the grounded chat-answer prompt (the fixed answer format).

The coordinator graph node calls these; tests call them directly with a fake
embedder.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.schemas import (
    AgentReport,
    ConsolidatedReport,
    Effort,
    Finding,
    Severity,
)

EmbedFn = Callable[[list[str]], np.ndarray]

ANSWER_SYSTEM_PROMPT = (
    "You are the lead reviewer for a multi-agent UI/UX audit, answering follow-up "
    "questions in a chat.\n"
    "Fixed answer format:\n"
    "1. Answer directly in 2-5 sentences of plain prose.\n"
    "2. Ground every claim in the audit findings or cited principles provided. If "
    "something was not assessed, say so.\n"
    "3. Name specific findings and principles (e.g. 'WCAG 1.4.3', 'Hick's Law').\n"
    "4. End with one line starting with 'Next:' giving the single most useful action.\n"
    "No JSON, no markdown headers."
)


# ----------------------------------------------------------------- dedup + score
def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def deduplicate(findings: list[Finding], embed: EmbedFn, threshold: float) -> list[Finding]:
    """Merge near-duplicate findings; keep the most urgent, union the citations."""
    if len(findings) <= 1:
        return findings
    try:
        vectors = _l2_normalize(embed([f"{f.title}. {f.description}" for f in findings]))
    except Exception:
        return findings  # degrade gracefully if embedding is unavailable

    assigned: set[int] = set()
    merged: list[Finding] = []
    for i in range(len(findings)):
        if i in assigned:
            continue
        group = [findings[i]]
        assigned.add(i)
        for j in range(i + 1, len(findings)):
            if j in assigned:
                continue
            if float(np.dot(vectors[i], vectors[j])) >= threshold:
                group.append(findings[j])
                assigned.add(j)
        merged.append(_merge(group))
    return merged


def _merge(group: list[Finding]) -> Finding:
    primary = min(group, key=lambda f: f.priority_score)
    seen: dict[str, object] = {}
    for f in group:
        for c in f.citations:
            seen.setdefault(c.kb_id, c)
    primary.citations = list(seen.values())  # type: ignore[assignment]
    primary.confidence = max(f.confidence for f in group)
    return primary


def overall_score(reports: list[AgentReport], findings: list[Finding]) -> int:
    ok = [r for r in reports if r.ok]
    if not ok:
        return 0
    agent_avg = sum(r.score for r in ok) / len(ok)
    penalty = sum(
        {"critical": 12, "high": 7, "medium": 3, "low": 1}[f.severity.value] for f in findings
    )
    score = 0.6 * agent_avg + 0.4 * max(0, 100 - penalty)
    return int(max(0, min(100, round(score))))


def prepare(
    reports: list[AgentReport], embed: EmbedFn, threshold: float
) -> tuple[list[Finding], list[Finding], int]:
    """Merge → prioritise → quick wins → score. The fast, local part."""
    findings = [f for r in reports if r.ok for f in r.findings]
    deduped = deduplicate(findings, embed, threshold)
    deduped.sort(key=lambda f: f.priority_score)
    quick_wins = [
        f for f in deduped
        if f.effort == Effort.LOW and f.severity in (Severity.CRITICAL, Severity.HIGH)
    ][:5]
    return deduped, quick_wins, overall_score(reports, deduped)


# --------------------------------------------------------------------- prompts
def summary_messages(findings: list[Finding], score: int) -> list:
    bullets = "\n".join(
        f"- [{f.severity.value}] {f.title} ({f.location}): {f.recommendation}"
        for f in findings[:12]
    ) or "No significant issues were detected."
    system = (
        "You are the lead design reviewer synthesising a multi-agent UI/UX audit. "
        "Write a concise executive summary (4-6 sentences) for a product team. Lead "
        "with the single most important theme, then the next priorities. Be specific "
        "and constructive. Plain prose only — no JSON, no headers."
    )
    user = f"Overall score: {score}/100.\nDe-duplicated findings:\n{bullets}"
    return [SystemMessage(content=system), HumanMessage(content=user)]


def report_context(consolidated: dict | None) -> str:
    if not consolidated:
        return "No findings are available yet."
    return ConsolidatedReport.model_validate(consolidated).to_markdown()


def answer_messages(question: str, consolidated: dict | None, history: list, grounding: str) -> list:
    system = (
        f"{ANSWER_SYSTEM_PROMPT}\n\n"
        f"=== AUDIT CONTEXT ===\n{report_context(consolidated)}\n\n"
        f"=== RELEVANT DESIGN PRINCIPLES ===\n{grounding}"
    )
    return [SystemMessage(content=system), *history[-8:], HumanMessage(content=question)]
