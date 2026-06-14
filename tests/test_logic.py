"""
Offline logic tests (no network/models). Run:
    PYTHONPATH=. python tests/test_logic.py
"""

from __future__ import annotations

import numpy as np

from src.agents import critique
from src.core.schemas import (
    AgentReport,
    Citation,
    ConsolidatedReport,
    Effort,
    Finding,
    Severity,
)


def mk(title, sev=Severity.HIGH, effort=Effort.LOW, conf=0.8):
    return Finding(title=title, description=f"{title} description", severity=sev,
                   recommendation=f"fix {title}", effort=effort, confidence=conf)


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(64, dtype=np.float32)
        for tok in t.lower().split():
            v[hash(tok) % 64] += 1.0
        out.append(v)
    return np.asarray(out, dtype=np.float32)


def test_priority_and_export():
    assert mk("a", Severity.CRITICAL).priority_score < mk("b", Severity.LOW).priority_score
    r = ConsolidatedReport(overall_score=80, executive_summary="Looks good.",
                           prioritised_findings=[mk("Contrast too low")])
    md = r.to_markdown()
    assert "Executive summary" in md and "Contrast too low" in md


def test_dedup_merges_and_unions_citations():
    f1 = mk("Primary CTA contrast is too low")
    f1.citations = [Citation(kb_id="wcag-143", title="Contrast", source="WCAG")]
    f2 = mk("Primary CTA contrast is too low")
    f2.citations = [Citation(kb_id="vis-contrast", title="Contrast discipline", source="Material")]
    f3 = mk("Navigation menu is overcrowded")
    merged = critique.deduplicate([f1, f2, f3], fake_embed, threshold=0.9)
    assert len(merged) == 2
    cta = next(m for m in merged if "CTA" in m.title)
    assert {c.kb_id for c in cta.citations} == {"wcag-143", "vis-contrast"}


def test_score_penalises_issues():
    clean = [AgentReport(agent="A", summary="", score=90, findings=[])]
    noisy = [AgentReport(agent="A", summary="", score=90,
                         findings=[mk("x", Severity.CRITICAL) for _ in range(5)])]
    assert critique.overall_score(clean, []) > critique.overall_score(noisy, noisy[0].findings)


def test_prepare_returns_quick_wins():
    reports = [AgentReport(agent="A", summary="", score=80,
                           findings=[mk("Low contrast CTA", Severity.CRITICAL, Effort.LOW)])]
    deduped, quick_wins, score = critique.prepare(reports, fake_embed, 0.9)
    assert len(deduped) == 1 and len(quick_wins) == 1 and 0 <= score <= 100


if __name__ == "__main__":
    test_priority_and_export()
    test_dedup_merges_and_unions_citations()
    test_score_penalises_issues()
    test_prepare_returns_quick_wins()
    print("All logic tests passed.")
