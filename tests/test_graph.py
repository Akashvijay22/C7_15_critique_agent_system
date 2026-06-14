"""
Graph integration test with fakes — no network, no real models, no embeddings.

Verifies the LangGraph wiring end to end:
- fan-out runs one agent per selected spec and the reports fan in,
- the coordinator produces a consolidated report,
- the graph pauses at the chat interrupt,
- resuming with Command(resume=question) answers and persists chat history.

Run: PYTHONPATH=. python tests/test_graph.py
"""

from __future__ import annotations

import numpy as np
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config import Settings
from src.core.schemas import AgentReport, Citation, Finding, Severity
from src.graph.builder import build_graph

USAGE_META = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


class FakeStructured:
    def __init__(self, include_raw=False):
        self._include_raw = include_raw

    def invoke(self, messages):
        report = AgentReport(
            agent="placeholder", summary="fake summary", score=80,
            findings=[Finding(
                title="Low contrast on CTA", description="hard to read",
                severity=Severity.HIGH, recommendation="raise contrast",
                citations=[Citation(kb_id="wcag-143", title="", source="")],  # backfilled
            )],
        )
        if self._include_raw:  # mirror with_structured_output(include_raw=True)
            return {"raw": AIMessage(content="", usage_metadata=USAGE_META),
                    "parsed": report, "parsing_error": None}
        return report


class FakeChat:
    def invoke(self, messages):
        return AIMessage(content="Grounded answer. Next: fix the CTA contrast.",
                         usage_metadata=USAGE_META)


class FakeFactory:
    def structured(self, model, schema, *, include_raw=False):
        return FakeStructured(include_raw=include_raw)

    def chat(self, model, **kwargs):
        return FakeChat()


class FakeKB:
    def context_block(self, query, category=None, k=None):
        return "[wcag-143] Contrast (Minimum)", [
            Citation(kb_id="wcag-143", title="WCAG 1.4.3 Contrast (Minimum)", source="WCAG 2.2")
        ]

    def embed(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32)


def build():
    settings = Settings(openrouter_api_key="x")
    return build_graph(FakeFactory(), FakeKB(), settings, checkpointer=InMemorySaver())


def test_analysis_then_interrupt_chat():
    graph = build()
    config = {"configurable": {"thread_id": "t1"}}
    initial = {
        "images": [{"b64": "eHg=", "mime": "image/png"}],
        "model": "m",
        "selected_agents": ["Visual Analysis Agent", "UX Critique Agent"],
        "extra_context": "",
        "reports": [],
        "messages": [],
    }
    graph.invoke(initial, config)  # runs to the chat interrupt and pauses

    snap = graph.get_state(config)
    values = snap.values
    assert len(values["reports"]) == 2                 # both agents fanned in
    assert values.get("analysis_done") is True
    assert values.get("consolidated") is not None

    # cost/latency: one usage record per agent + the coordinator, tokens captured
    usage = values.get("usage", [])
    assert len(usage) == 3                              # 2 agents + coordinator
    assert all(u["input_tokens"] == 10 and u["output_tokens"] == 20 for u in usage)
    assert all("latency" in u and u["cost"] >= 0 for u in usage)
    # graph is paused awaiting a question
    assert snap.next == ("chat",) or len(snap.interrupts) >= 1

    # citation was backfilled from the empty-title stub
    report = values["consolidated"]
    titles = {c["title"] for f in report["prioritised_findings"] for c in f["citations"]}
    assert "WCAG 1.4.3 Contrast (Minimum)" in titles

    # resume with a question → answered and persisted
    graph.invoke(Command(resume="Why is the contrast a problem?"), config)
    msgs = graph.get_state(config).values["messages"]
    assert len(msgs) == 2
    assert msgs[0].content == "Why is the contrast a problem?"
    assert "Next:" in msgs[1].content


if __name__ == "__main__":
    test_analysis_then_interrupt_chat()
    print("Graph wiring + interrupt test passed.")
