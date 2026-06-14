"""
Redesign agent — opt-in, runs *outside* the LangGraph StateGraph.

Given a completed critique, ask an image-output model to produce a single
improved, high-fidelity mockup of the original design. It is an **AI concept for
direction**, not production art.

Defining properties (see SPEC_REDESIGN_AND_METRICS.md, Part A):
- **Trigger:** on-demand only, when the user clicks "Generate improved design".
- **Not a graph node:** a plain function call after a report exists.
- **Model:** ``settings.redesign_model`` (env ``REDESIGN_MODEL``), an image-output model.
- **Input:** the primary design only (``images[:1]``).
- **Cost:** OpenRouter's exact ``usage["cost"]`` when present, else a token estimate.
- **Failure mode:** never raises to the UI — failures come back as ``error`` in the
  returned dict (rendered as ``st.error``); an empty image list is a soft warning.

This module has no Streamlit dependency so it stays unit-testable.
"""

from __future__ import annotations

import logging
import time

from config import Settings, estimate_cost
from src.core import llm

logger = logging.getLogger(__name__)

# Bounds that keep the prompt (and therefore the cost) predictable. See A4.
MAX_PRIORITISED = 8        # at most this many prioritised fixes
MAX_RECS_PER_AGENT = 2     # at most this many recommendations per specialist
MAX_RECS_TOTAL = 6         # ...capped at this many specialist recs overall

# A complete HTML document is long; give the model plenty of output headroom so
# the result isn't cut off mid-tag (the common cause of a broken render).
HTML_MAX_TOKENS = 12000


def _review_brief(report: dict, mode: str = "single") -> str:
    """The shared, bounded review summary fed to every redesign mode (A4).

    Order: prioritised fixes (≤8) → specialist recs (≤2/agent, ≤6 total) →
    compare-mode steer. All slices are intentional caps to keep cost predictable.
    """
    lines: list[str] = []

    # 1) Prioritised fixes — the coordinator's ranked findings.
    actions = []
    for finding in (report.get("prioritised_findings") or [])[:MAX_PRIORITISED]:
        action = finding.get("recommendation") or finding.get("title")
        if action:
            actions.append(action.strip())
    if actions:
        lines.append("Prioritised fixes to apply:")
        lines.extend(f"- {a}" for a in actions)

    # 2) Specialist recommendations — ≤2 per agent, capped at 6 total, tagged by lens.
    recs: list[str] = []
    for agent_report in report.get("agent_reports") or []:
        if agent_report.get("error") is not None:  # skip agents that failed
            continue
        lens = agent_report.get("agent", "agent")
        for finding in (agent_report.get("findings") or [])[:MAX_RECS_PER_AGENT]:
            rec = finding.get("recommendation") or finding.get("title")
            if rec:
                recs.append(f"- ({lens}) {rec.strip()}")
            if len(recs) >= MAX_RECS_TOTAL:
                break
        if len(recs) >= MAX_RECS_TOTAL:
            break
    if recs:
        lines.append("\nSpecialist recommendations:")
        lines.extend(recs)

    # 3) Compare-mode steer — base the redesign on the overall verdict, merge the best.
    if mode == "compare":
        summary = (report.get("executive_summary") or "").strip()
        if summary:
            lines.append(
                "\nThis was a comparison of several designs. Base the redesign on the "
                f"overall recommendation and merge the best of the others:\n{summary}"
            )
    return "\n".join(lines)


def build_redesign_prompt(report: dict, mode: str = "single") -> str:
    """Assemble the image-generation redesign prompt: role → brief → constraints."""
    return (
        "You are a senior product designer. Improve the attached UI screen into a "
        "polished, production-quality mockup that resolves the issues below while "
        "keeping the same content, brand, and intent.\n\n"
        f"{_review_brief(report, mode)}\n\n"
        "Constraints: preserve the existing content, branding, and intent; improve "
        "visual hierarchy, contrast and accessibility, spacing, and typography; make "
        "the primary call-to-action prominent. Output a single clean, modern, "
        "high-fidelity image."
    )


def build_html_prompt(report: dict, mode: str = "single") -> str:
    """Assemble the HTML redesign prompt: role → brief → output constraints.

    Unlike the image prompt, this asks for a single self-contained HTML document
    so the result has real, selectable text and renders live in the browser.
    """
    return (
        "You are a senior product designer and front-end engineer. Recreate the "
        "attached UI screen as an improved, high-fidelity mockup, applying the fixes "
        "below while preserving the same content, copy, and branding.\n\n"
        f"{_review_brief(report, mode)}\n\n"
        "Output requirements:\n"
        "- Return ONE complete, self-contained HTML document (<!doctype html> … </html>).\n"
        "- All CSS inline in a single <style> tag; no external files, scripts, or CDN links.\n"
        "- Use real text from the screen — do not use lorem ipsum or placeholder gibberish.\n"
        "- Improve visual hierarchy, contrast and accessibility (WCAG AA), spacing, and "
        "typography; make the primary call-to-action clearly dominant.\n"
        "- Make it responsive and use system fonts.\n"
        "- Output ONLY the HTML. No explanations, no markdown code fences."
    )


def _extract_html(text: str) -> str:
    """Strip any ```html fences the model may wrap the document in."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: s.rfind("```")]
    return s.strip()


def run_redesign(
    report: dict,
    images: list[dict],
    settings: Settings,
    *,
    mode: str = "single",
) -> dict:
    """Generate an improved mockup from a completed review.

    ``report`` is a ``ConsolidatedReport`` dict; ``images`` are the app's image
    refs (``{"b64", "mime", ...}``) — only the primary design (``images[:1]``)
    is sent, even in compare mode.

    Never raises: any failure is returned as the ``error`` field. Returns::

        {"images": list[bytes], "text": str, "model": str, "cost": float,
         "prompt": str, "latency": float, "error": str | None}
    """
    model = settings.redesign_model
    prompt = build_redesign_prompt(report, mode=mode)
    primary = [{"b64": im["b64"], "mime": im["mime"]} for im in (images or [])[:1]]

    started = time.perf_counter()
    try:
        out_images, text, usage = llm.generate_image(settings, model, prompt, primary)
        error = None
    except Exception as exc:  # never crash the app — surface as error in the UI
        logger.exception("Redesign generation failed")
        out_images, text, usage, error = [], "", {}, str(exc)
    latency = time.perf_counter() - started

    # Cost: prefer OpenRouter's exact image-gen cost, else fall back to a token estimate (B3).
    cost = usage.get("cost")
    if cost is None:
        cost = estimate_cost(
            model,
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
        )

    return {
        "images": out_images,
        "text": text,
        "model": model,
        "cost": float(cost),
        "prompt": prompt,
        "latency": latency,   # A6: redesign now reports latency like every other agent
        "error": error,
    }


def run_redesign_html(factory, model: str, report: dict, images: list[dict],
                      *, mode: str = "single") -> dict:
    """Generate an improved mockup as a self-contained HTML document.

    Uses the vision chat model (so it can match the original layout) via the
    shared ``LLMFactory`` — HTML is just text, so this goes through ``ChatOpenAI``
    rather than the image transport. Only the primary design (``images[:1]``) is sent.

    Never raises: failures come back in ``error``. Returns::

        {"html": str, "model": str, "cost": float, "prompt": str,
         "latency": float, "error": str | None}
    """
    prompt = build_html_prompt(report, mode=mode)
    primary = [{"b64": im["b64"], "mime": im["mime"]} for im in (images or [])[:1]]
    system = "You are a senior product designer and front-end engineer."

    in_tok = out_tok = 0
    html = ""
    started = time.perf_counter()
    try:
        messages = llm.multimodal_messages(system, prompt, primary)
        # A full HTML document easily exceeds the default cap, so request a large
        # output budget — truncation is the usual cause of a broken render.
        resp = factory.chat(model, streaming=False, max_tokens=HTML_MAX_TOKENS).invoke(messages)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        html = _extract_html(content)
        in_tok, out_tok = llm.usage_from_message(resp)
        error = None
    except Exception as exc:  # never crash the app — surface as error in the UI
        logger.exception("HTML redesign generation failed")
        error = str(exc)
    latency = time.perf_counter() - started

    return {
        "html": html,
        "model": model,
        "cost": estimate_cost(model, in_tok, out_tok),
        "prompt": prompt,
        "latency": latency,
        "truncated": bool(html) and "</html>" not in html.lower(),
        "error": error,
    }
