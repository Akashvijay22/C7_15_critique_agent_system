"""
Graph state.

``reports`` uses the ``operator.add`` reducer so the parallel agent nodes
(fanned out with ``Send``) can each append their result without clobbering one
another — the standard LangGraph map-reduce pattern. ``messages`` uses
``add_messages`` so the chat turns accumulate across interrupts; combined with a
checkpointer this is the durable session memory.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ImageRef(TypedDict):
    b64: str
    mime: str


class CritiqueState(TypedDict, total=False):
    # inputs
    images: list[ImageRef]
    model: str
    selected_agents: list[str]
    extra_context: str
    # analysis (fan-in)
    reports: Annotated[list[dict], operator.add]
    # per-node cost/latency records, merged across parallel nodes (see nodes.py)
    usage: Annotated[list[dict], operator.add]
    consolidated: Optional[dict]
    analysis_done: bool
    # chat / session memory
    messages: Annotated[list[AnyMessage], add_messages]


class AgentTask(TypedDict):
    """The custom per-agent state delivered by a Send during fan-out."""

    spec_name: str
    images: list[ImageRef]
    model: str
    extra_context: str
