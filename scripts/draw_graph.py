"""
Render the compiled LangGraph to an image + Mermaid text.

``build_graph`` only *wires* the nodes (the factory/KB are captured in closures and
never invoked at build time), so we can compile with lightweight stand-ins — no
API key, LanceDB, or embedding model required.

Run:  PYTHONPATH=. python scripts/draw_graph.py
Outputs: docs/graph.png (best effort) and docs/graph.mmd (always).
"""

from __future__ import annotations

import os

from config import Settings
from src.graph.builder import build_graph


class _Stub:
    """Stands in for LLMFactory / KnowledgeBase — only identity is needed to wire."""


def main() -> None:
    graph = build_graph(_Stub(), _Stub(), Settings(openrouter_api_key="x"))
    drawable = graph.get_graph()

    os.makedirs("docs", exist_ok=True)

    mermaid = drawable.draw_mermaid()
    with open("docs/graph.mmd", "w", encoding="utf-8") as fh:
        fh.write(mermaid)
    print("Wrote docs/graph.mmd")

    try:
        png = drawable.draw_mermaid_png()  # uses mermaid.ink; needs network
        with open("docs/graph.png", "wb") as fh:
            fh.write(png)
        print("Wrote docs/graph.png")
    except Exception as exc:  # offline / no renderer — Mermaid text is the fallback
        print(f"Could not render PNG ({exc}); use docs/graph.mmd instead.")

    print("\n--- Mermaid ---\n" + mermaid)


if __name__ == "__main__":
    main()
