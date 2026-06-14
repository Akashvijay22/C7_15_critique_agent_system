"""
Graph builder.

Wires the StateGraph:

    START ─(dispatch: Send×N)─▶ agent ─▶ coordinator ─▶ chat ⇄ chat (interrupt loop)

- ``dispatch`` fans out one ``agent`` task per selected agent (map).
- The static edge ``agent → coordinator`` makes the coordinator a fan-in barrier:
  it runs once, after every parallel agent task in the superstep completes.
- ``coordinator → chat`` enters a self-looping chat node that ``interrupt()``s to
  await each question and resumes on ``Command(resume=...)``.

Compiled with a checkpointer, the whole state (reports, consolidated report, and
chat history) persists per ``thread_id`` — durable session memory and resumable
interrupts, for free.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from config import Settings
from src.core.llm import LLMFactory
from src.graph.nodes import (
    dispatch,
    make_agent_node,
    make_chat_node,
    make_coordinator_node,
)
from src.graph.state import CritiqueState
from src.rag.knowledge_base import KnowledgeBase


def build_graph(
    factory: LLMFactory,
    kb: KnowledgeBase,
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile and return the critique graph."""
    builder = StateGraph(CritiqueState)

    builder.add_node("agent", make_agent_node(factory, kb))
    builder.add_node("coordinator", make_coordinator_node(factory, kb, settings))
    builder.add_node("chat", make_chat_node(factory, kb))

    # Map: fan out from START to one `agent` task per selected agent.
    builder.add_conditional_edges(START, dispatch, ["agent"])
    # Reduce: all agent tasks converge on the coordinator.
    builder.add_edge("agent", "coordinator")
    # Then enter the chat loop (pauses on interrupt each turn).
    builder.add_edge("coordinator", "chat")
    builder.add_edge("chat", "chat")

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
