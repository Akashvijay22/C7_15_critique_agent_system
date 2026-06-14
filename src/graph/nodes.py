"""
Graph nodes.

Built as closures over the shared ``LLMFactory`` and ``KnowledgeBase`` so the
node functions stay pure-ish and the dependencies are injected once in
``builder.build_graph`` (and faked in tests).

Nodes:
- ``agent_node``       — runs one specialist (structured output), appends a report.
- ``coordinator_node`` — fan-in: dedup/score, then a streamed executive summary.
- ``chat_node``        — interrupt()s to await a user question, answers it grounded
                         in the report + history, then loops to await the next.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send, interrupt

from config import Settings, estimate_cost
from src.agents import critique
from src.agents.specs import SPECS
from src.core.llm import LLMFactory, multimodal_messages, usage_from_message
from src.core.schemas import AgentReport, ConsolidatedReport, Citation
from src.graph.state import CritiqueState
from src.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


def _usage_record(agent: str, label: str, model: str, in_tok: int, out_tok: int,
                  latency: float) -> dict:
    """One row for the per-run cost & latency table."""
    return {
        "agent": agent,
        "label": label,
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost": estimate_cost(model, in_tok, out_tok),
        "latency": round(latency, 3),
    }


def dispatch(state: CritiqueState) -> list[Send]:
    """Map step: one parallel ``agent`` task per selected agent."""
    selected = state.get("selected_agents") or list(SPECS.keys())
    return [
        Send("agent", {
            "spec_name": name,
            "images": state["images"],
            "model": state["model"],
            "extra_context": state.get("extra_context", ""),
        })
        for name in selected if name in SPECS
    ]


def make_agent_node(factory: LLMFactory, kb: KnowledgeBase) -> Callable:
    def agent_node(task: dict) -> dict:
        spec = SPECS[task["spec_name"]]
        model = task["model"]
        in_tok = out_tok = 0
        latency = 0.0
        try:
            grounding, citations = kb.context_block(spec.retrieval_query, category=spec.category)
            ctx = task.get("extra_context", "")
            user_text = (
                f"Critique the attached screen(s) through your lens ({spec.category})."
                + (f"\n\nUser context: {ctx}" if ctx.strip() else "")
            )
            messages = multimodal_messages(spec.system_prompt(grounding), user_text, task["images"])
            # include_raw=True so we get token usage off the raw message for the cost table.
            started = time.perf_counter()
            result = factory.structured(model, AgentReport, include_raw=True).invoke(messages)
            latency = time.perf_counter() - started
            report, in_tok, out_tok = _unpack_structured(result)
            report.agent = spec.name
            report.model = model
            _backfill_citations(report, citations)
        except Exception as exc:  # one agent failing must not sink the graph
            logger.exception("%s failed", spec.name)
            report = AgentReport(
                agent=spec.name, summary="This agent could not complete its analysis.",
                score=0, findings=[], model=model, error=str(exc),
            )
        usage = _usage_record(spec.name, spec.name, model, in_tok, out_tok, latency)
        return {"reports": [report.model_dump(mode="json")], "usage": [usage]}

    return agent_node


def _unpack_structured(result) -> tuple[AgentReport, int, int]:
    """Split a ``with_structured_output(include_raw=True)`` result.

    Tolerates fakes/older runnables that return the parsed model directly (then
    there is no usage to read, so tokens are ``0``). Raises if parsing failed so
    the caller falls back to the error report.
    """
    if isinstance(result, dict):
        report = result.get("parsed")
        if report is None:
            raise ValueError(f"structured output failed to parse: {result.get('parsing_error')}")
        in_tok, out_tok = usage_from_message(result.get("raw"))
        return report, in_tok, out_tok
    return result, 0, 0


def make_coordinator_node(factory: LLMFactory, kb: KnowledgeBase, settings: Settings) -> Callable:
    def coordinator_node(state: CritiqueState) -> dict:
        reports = [AgentReport.model_validate(r) for r in state.get("reports", [])]
        deduped, quick_wins, overall = critique.prepare(
            reports, kb.embed, settings.dedup_similarity_threshold
        )
        model = state["model"]
        in_tok = out_tok = 0
        latency = 0.0
        # Streamed via stream_mode="messages" because the model has streaming on.
        try:
            started = time.perf_counter()
            resp = factory.chat(model).invoke(critique.summary_messages(deduped, overall))
            latency = time.perf_counter() - started
            in_tok, out_tok = usage_from_message(resp)
            summary = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            summary = (
                f"The design scored {overall}/100. "
                + (f"The most pressing issue is '{deduped[0].title}'." if deduped
                   else "No significant issues were detected.")
            )
        consolidated = ConsolidatedReport(
            overall_score=overall, executive_summary=summary.strip(),
            quick_wins=quick_wins, prioritised_findings=deduped, agent_reports=reports,
        )
        usage = _usage_record("coordinator", "Coordinator (synthesis)", model,
                              in_tok, out_tok, latency)
        return {"consolidated": consolidated.model_dump(mode="json"),
                "usage": [usage], "analysis_done": True}

    return coordinator_node


def make_chat_node(factory: LLMFactory, kb: KnowledgeBase) -> Callable:
    def chat_node(state: CritiqueState) -> dict:
        # Pause the graph and wait for the user's question (human-in-the-loop).
        payload = interrupt({"awaiting": "question"})
        question = payload if isinstance(payload, str) else (payload or {}).get("question", "")
        if not question:
            return {}
        try:
            grounding, _ = kb.context_block(question)
        except Exception:
            grounding = ""
        messages = critique.answer_messages(
            question, state.get("consolidated"), state.get("messages", []), grounding
        )
        resp = factory.chat(state["model"]).invoke(messages)
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"messages": [HumanMessage(content=question), AIMessage(content=answer)]}

    return chat_node


def _backfill_citations(report: AgentReport, available: list[Citation]) -> None:
    by_id = {c.kb_id: c for c in available}
    for finding in report.findings:
        finding.citations = [by_id[c.kb_id] for c in finding.citations if c.kb_id in by_id]
