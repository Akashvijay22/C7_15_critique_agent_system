"""
Multimodal AI Design Analysis Suite — Streamlit front end (LangChain + LangGraph).

Run with:  streamlit run app.py

Inputs: upload multiple screenshots and/or paste straight from the clipboard.
Orchestration is a LangGraph StateGraph (src/graph). This file streams the
analysis, renders the consolidated report, draws an annotated copy of each
screen (boxes + numbered pins + a comments legend, downloadable), and drives the
interrupt-based chat. The graph's checkpointer holds session context per thread.
"""

from __future__ import annotations

import base64
import hashlib
import io
import uuid

import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command

try:
    from streamlit_paste_button import paste_image_button as paste_button
except Exception:  # component optional — app still works via upload
    paste_button = None

from config import MODELS_BY_ID, SUPPORTED_MODELS, Settings, settings as base_settings
from src.agents.redesign import run_redesign, run_redesign_html
from src.agents.specs import available_agents
from src.core.annotate import annotate_image
from src.core.llm import LLMFactory
from src.core.schemas import ConsolidatedReport, Finding
from src.graph.builder import build_graph
from src.rag.knowledge_base import KnowledgeBase

st.set_page_config(page_title="Multimodal UI/UX Critique Suite",
                   page_icon=":material/design_services:", layout="wide")

SEVERITY_COLORS = {"critical": "#ff4d6d", "high": "#ff9f43", "medium": "#ffd166", "low": "#6c9cff"}
MIME_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}

# Theme tokens that differ between dark and light. Everything else in _CSS_BODY
# is written against these CSS variables, so flipping the mode only swaps :root.
_DARK_TOKENS = {
    "--text": "#e7eaf3",
    "--muted": "#9aa3b8",
    "--surface": "rgba(255,255,255,0.035)",
    "--surface-strong": "rgba(255,255,255,0.06)",
    "--border": "rgba(255,255,255,0.09)",
    "--sidebar-bg": "linear-gradient(180deg, rgba(20,26,42,0.92), rgba(13,17,28,0.92))",
    "--bg": ("radial-gradient(1100px 600px at 12% -8%, rgba(124,156,255,0.16), transparent 60%),"
             "radial-gradient(900px 600px at 95% 0%, rgba(176,108,255,0.14), transparent 55%),"
             "linear-gradient(180deg, #0b0f19 0%, #0a0d16 100%)"),
}
_LIGHT_TOKENS = {
    "--text": "#1b2333",
    "--muted": "#5b6473",
    "--surface": "rgba(17,24,39,0.035)",
    "--surface-strong": "rgba(17,24,39,0.06)",
    "--border": "rgba(17,24,39,0.12)",
    "--sidebar-bg": "linear-gradient(180deg, #ffffff, #eef1f8)",
    "--bg": ("radial-gradient(1100px 600px at 12% -8%, rgba(124,156,255,0.10), transparent 60%),"
             "radial-gradient(900px 600px at 95% 0%, rgba(176,108,255,0.08), transparent 55%),"
             "linear-gradient(180deg, #f7f8fc 0%, #eef1f8 100%)"),
}

# Theme-agnostic styling — all colours come from the :root variables above.
_CSS_BODY = """
/* Tailwind CSS default `font-sans` stack (the typical React/Tailwind look) */
html, body, [class*="css"], .stApp,
button, input, textarea, select {
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif,
    "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
}

/* Ambient gradient backdrop + base text colour */
.stApp { background: var(--bg); background-attachment: fixed; color: var(--text); }
.stApp p, .stApp li, .stApp span, .stMarkdown,
[data-testid="stWidgetLabel"] label, [data-testid="stMetricLabel"],
.stRadio label, .stCheckbox label, h2, h3 { color: var(--text); }
[data-testid="stCaptionContainer"], small { color: var(--muted) !important; }

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
/* Gradient only on main-content h1 (not the sidebar — clipping eats emoji color) */
section.main h1, [data-testid="stAppViewContainer"] .main h1 {
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
/* Keep the sidebar title as solid text so its emoji icon stays visible */
[data-testid="stSidebar"] h1 {
  background: none; -webkit-text-fill-color: initial; color: var(--text);
}

/* Sidebar as a glass panel */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  backdrop-filter: blur(8px);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid var(--border);
  background: var(--surface-strong); color: var(--text); font-weight: 600;
  transition: transform .12s ease, box-shadow .2s ease, border-color .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); border-color: var(--accent);
  box-shadow: 0 8px 24px rgba(124,156,255,0.18);
}
button[kind="primary"], button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
  border: none !important; color: #0b0f19 !important; font-weight: 700 !important;
  box-shadow: 0 8px 22px rgba(124,156,255,0.30) !important;
}
button[kind="primary"]:hover { filter: brightness(1.06); transform: translateY(-1px); }

/* Cards: metrics, expanders, status, chat messages */
[data-testid="stMetric"],
[data-testid="stExpander"] details,
[data-testid="stStatusWidget"],
[data-testid="stNotification"],
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(6px);
}
[data-testid="stMetric"] { padding: 14px 18px; }
[data-testid="stMetricValue"] { color: var(--accent); font-weight: 700; }
[data-testid="stExpander"] details { padding: 2px 6px; }
[data-testid="stExpander"] summary:hover { color: var(--accent); }

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
  border-radius: 10px; padding: 6px 14px; background: var(--surface); border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(124,156,255,0.22), rgba(176,108,255,0.22));
  border-color: var(--accent);
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stChatInput textarea, [data-baseweb="select"] > div {
  border-radius: 12px !important; border: 1px solid var(--border) !important;
  background: var(--surface) !important; color: var(--text) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--accent) !important; }

/* Tables, images, dividers */
[data-testid="stTable"] table { border-radius: 12px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--border); }
hr { border-color: var(--border); }

/* Slim scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(124,156,255,0.35); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,156,255,0.55); }
"""


def apply_theme(dark: bool = True) -> None:
    """Inject the sleek CSS layer for the chosen mode (dark or light) at runtime."""
    tokens = _DARK_TOKENS if dark else _LIGHT_TOKENS
    root = ":root {\n  --accent: #7c9cff;\n  --accent-2: #b06cff;\n"
    root += "".join(f"  {k}: {v};\n" for k, v in tokens.items())
    root += "}"
    st.markdown(f"<style>\n{root}\n{_CSS_BODY}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading design knowledge base…")
def get_kb(lancedb_path: str, embedding_model: str) -> KnowledgeBase:
    return KnowledgeBase(Settings(lancedb_path=lancedb_path, embedding_model=embedding_model))


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("phase", "setup")
    ss.setdefault("graph", None)
    ss.setdefault("thread_id", None)
    ss.setdefault("pasted", [])         # [{b64, mime, name}]
    ss.setdefault("paste_hashes", set())
    ss.setdefault("redesign", None)     # cached output of the redesign agent
    ss.setdefault("dark_mode", False)   # UI theme toggle (light by default)


def reset() -> None:
    for k in ("phase", "graph", "thread_id", "pasted", "paste_hashes", "redesign"):
        st.session_state.pop(k, None)
    init_state()


# --------------------------------------------------------------------- sidebar
def sidebar() -> dict:
    st.sidebar.title(":material/design_services: Critique Suite")
    st.sidebar.caption("LangGraph multi-agent · streaming · interruptible")
    # Theme toggle — flips the injected CSS on the next rerun (see apply_theme).
    st.sidebar.toggle("Dark mode", key="dark_mode",
                      help="Switch the interface between dark and light.")
    api_key = st.sidebar.text_input("OpenRouter API key", value=base_settings.openrouter_api_key,
                                    type="password", help="From openrouter.ai/keys.")
    ids = [m.id for m in SUPPORTED_MODELS]
    idx = ids.index(base_settings.default_model) if base_settings.default_model in ids else 0
    model = st.sidebar.selectbox("Model", ids, index=idx, format_func=lambda m: MODELS_BY_ID[m].label)
    st.sidebar.caption(MODELS_BY_ID[model].notes)
    agents = st.sidebar.multiselect("Agents", available_agents(), default=available_agents())
    with st.sidebar.expander("Advanced"):
        retrieval_k = st.slider("Grounding principles / agent", 2, 8, base_settings.retrieval_k)
        temperature = st.slider("Temperature", 0.0, 1.0, base_settings.temperature, 0.05)
        dedup = st.slider("Dedup threshold", 0.70, 0.95, base_settings.dedup_similarity_threshold, 0.01)
    if st.session_state.phase != "setup":
        st.sidebar.divider()
        if st.sidebar.button("New analysis", icon=":material/refresh:", use_container_width=True):
            reset(); st.rerun()
    return {"api_key": api_key, "model": model, "agents": agents,
            "retrieval_k": retrieval_k, "temperature": temperature, "dedup": dedup}


# --------------------------------------------------------------------- inputs
def collect_images() -> list[dict]:
    """Gather images from the uploader and the clipboard-paste button."""
    ss = st.session_state
    c1, c2 = st.columns([3, 1])
    with c1:
        uploads = st.file_uploader("Upload UI screenshots (multiple allowed)",
                                   type=list(MIME_BY_EXT.keys()), accept_multiple_files=True)
    with c2:
        st.caption("…or paste from clipboard")
        if paste_button is not None:
            res = paste_button("Paste screenshot")
            if getattr(res, "image_data", None) is not None:
                buf = io.BytesIO()
                res.image_data.convert("RGB").save(buf, "PNG")
                data = buf.getvalue()
                h = hashlib.md5(data).hexdigest()
                if h not in ss.paste_hashes:        # dedupe across reruns
                    ss.paste_hashes.add(h)
                    ss.pasted.append({"b64": base64.b64encode(data).decode(),
                                      "mime": "image/png", "name": f"pasted-{len(ss.pasted) + 1}.png"})
        else:
            st.info("Install `streamlit-paste-button` to enable clipboard paste.")

    images: list[dict] = []
    for u in uploads or []:
        ext = u.name.rsplit(".", 1)[-1].lower()
        images.append({"b64": base64.b64encode(u.getvalue()).decode(),
                       "mime": MIME_BY_EXT.get(ext, "image/png"), "name": u.name})
    images += ss.pasted

    if images:
        st.caption(f"{len(images)} image(s) ready")
        cols = st.columns(min(5, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(base64.b64decode(img["b64"]), caption=f"[{i}] {img['name']}", use_container_width=True)
        if ss.pasted and st.button("Clear image", icon=":material/delete:"):
            # Keep paste_hashes: the paste component re-returns the last image on
            # rerun, so dropping its hash would immediately re-add it.
            ss.pasted = []; st.rerun()
    return images


# --------------------------------------------------------------------- rendering
def render_finding(f: Finding, index: int | None = None) -> None:
    color = SEVERITY_COLORS.get(f.severity.value, "#888")
    header = f.title if index is None else f"{index}. {f.title}"
    st.markdown(
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:6px;"
        f"font-size:0.75rem;font-weight:600'>{f.severity.value.upper()}</span> &nbsp;**{header}**",
        unsafe_allow_html=True)
    loc = f.location + (f"  ·  :material/image: screen {f.region.image_index}" if f.region else "")
    st.caption(f":material/place: {loc}  ·  effort: {f.effort.value}  ·  confidence: {f.confidence:.0%}")
    st.write(f.description)
    st.markdown(f"**→ Recommendation:** {f.recommendation}")
    if f.citations:
        chips = "  ".join((f"[{c.title}]({c.url})" if c.url else c.title) + f" — *{c.source}*"
                          for c in f.citations)
        st.markdown(f"<small>Grounded in: {chips}</small>", unsafe_allow_html=True)
    st.divider()


def render_annotated(images: list[dict], report: ConsolidatedReport) -> None:
    """Mark each screen with the findings that point at it + a comments legend."""
    if not images:
        return
    st.subheader(":material/push_pin: Annotated screens")
    st.caption("Numbered pins mark where each issue is; the legend lists the fix. "
               "Findings without a located region are listed under the image.")
    numbered = list(enumerate(report.prioritised_findings, 1))
    tabs = st.tabs([f"Screen {i} · {img['name']}" for i, img in enumerate(images)])
    for i, (tab, img) in enumerate(zip(tabs, images)):
        with tab:
            items, unplaced = [], []
            for n, f in numbered:
                idx = f.region.image_index if f.region else 0
                if f.region and idx == i:
                    items.append({"number": n, "severity": f.severity.value, "title": f.title,
                                  "recommendation": f.recommendation,
                                  "region": {"x": f.region.x, "y": f.region.y,
                                             "w": f.region.w, "h": f.region.h}})
                elif not f.region and i == 0:
                    unplaced.append((n, f))
            raw = base64.b64decode(img["b64"])
            try:
                png = annotate_image(raw, items, with_legend=bool(items))
            except Exception as exc:  # never let drawing break the report
                st.warning(f"Could not annotate this image: {exc}")
                png = raw
            st.image(png, use_container_width=True)
            st.download_button(f"Download annotated screen {i}", data=png,
                               file_name=f"annotated_screen_{i}.png", mime="image/png",
                               icon=":material/download:", key=f"dl_{i}")
            if unplaced:
                st.markdown("**Not localised to a region:**")
                for n, f in unplaced:
                    st.markdown(f"- **{n}. {f.title}** ({f.severity.value}) — {f.recommendation}")


def render_report(report: ConsolidatedReport, images: list[dict]) -> None:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Overall score", f"{report.overall_score}/100")
        crit = sum(1 for f in report.prioritised_findings if f.severity.value == "critical")
        high = sum(1 for f in report.prioritised_findings if f.severity.value == "high")
        st.metric("Critical / High", f"{crit} / {high}")
    with c2:
        st.subheader("Executive summary")
        st.write(report.executive_summary)
    if report.quick_wins:
        st.subheader(":material/bolt: Quick wins")
        for f in report.quick_wins:
            st.markdown(f"- **{f.title}** — {f.recommendation}")

    render_annotated(images, report)

    tp, ta = st.tabs(["Prioritised findings", "By agent"])
    with tp:
        if not report.prioritised_findings:
            st.success("No significant issues detected.")
        for i, f in enumerate(report.prioritised_findings, 1):
            render_finding(f, i)
    with ta:
        for ar in report.agent_reports:
            label = f"{ar.agent} — {ar.score}/100" if ar.ok else f"{ar.agent} — :material/error: error"
            with st.expander(label):
                if ar.error:
                    st.error(ar.error); continue
                st.caption(ar.summary)
                for f in ar.findings:
                    render_finding(f)
    st.download_button("Download report (Markdown)", data=report.to_markdown(),
                       file_name="ui_critique_report.md", mime="text/markdown",
                       icon=":material/download:")


def render_cost_table(usage: list[dict]) -> None:
    """One row per agent: Agent · Model · Tokens · Latency (s) · Cost ($).

    Cost is estimated from token usage (approximate, not billing); latency is the
    wall-clock of each node's model call. Because the specialist nodes run in
    parallel, their latencies overlap and do **not** sum to wall-clock time.
    """
    if not usage:
        return
    with st.expander("Cost & latency", icon=":material/payments:"):
        rows = []
        total_tokens = total_cost = 0
        for u in usage:
            tokens = int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
            cost = float(u.get("cost", 0.0))
            total_tokens += tokens
            total_cost += cost
            rows.append({
                "Agent": u.get("label", u.get("agent", "?")),
                "Model": u.get("model", ""),
                "Tokens": tokens,
                "Latency (s)": round(float(u.get("latency", 0.0)), 2),
                "Cost ($)": round(cost, 5),
            })
        rows.append({"Agent": "Total (estimated)", "Model": "", "Tokens": total_tokens,
                     "Latency (s)": None, "Cost ($)": round(total_cost, 4)})
        st.table(rows)
        st.caption("Cost is an approximate estimate from token usage, not a bill. "
                   "Parallel agents' latencies overlap, so they don't sum to wall-clock.")


def render_redesign(data: dict) -> None:
    """Show the redesign agent's generated mockup (or its error/empty hint)."""
    if data.get("error"):
        st.error(f"Redesign failed: {data['error']}")
        return
    images = data.get("images") or []
    if not images:
        st.warning("The model returned no image. Try again or pick a different "
                   "REDESIGN_MODEL (it must be an image-output model).")
        return
    for i, raw in enumerate(images):
        st.image(raw, use_container_width=True, caption="AI redesign concept — for direction, not production art")
        st.download_button("Download redesign", data=raw, file_name=f"redesign_{i}.png",
                           mime="image/png", icon=":material/download:", key=f"dl_redesign_{i}")
    if data.get("text"):
        st.caption(data["text"])
    st.caption(f"Model: `{data.get('model','')}`  ·  Generation cost ≈ ${data.get('cost', 0.0):.4f}"
               f"  ·  Latency {data.get('latency', 0.0):.2f}s")


def render_redesign_html(data: dict) -> None:
    """Render the HTML redesign live, with code view + download."""
    if data.get("error"):
        st.error(f"Redesign failed: {data['error']}")
        return
    html = data.get("html") or ""
    if not html:
        st.warning("The model returned no HTML. Try again or pick a different model.")
        return
    if data.get("truncated"):
        st.warning("The HTML looks truncated (no closing </html>), so it may render "
                   "incompletely. Try generating again, or pick a model with a larger output limit.")
    components.html(html, height=1000, scrolling=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.download_button("Download HTML", data=html, file_name="redesign.html",
                           mime="text/html", icon=":material/download:", key="dl_redesign_html")
    with st.expander("View HTML source"):
        st.code(html, language="html")
    st.caption(f"Model: `{data.get('model','')}`  ·  Generation cost ≈ ${data.get('cost', 0.0):.4f}"
               f"  ·  Latency {data.get('latency', 0.0):.2f}s")


def render_redesign_tab(cfg: dict, consolidated: dict, images: list[dict]) -> None:
    """The 'Improved design' tab: format picker, trigger, and the rendered mockup."""
    st.caption("Generate a high-fidelity mockup that applies the prioritised fixes. "
               "An AI concept for direction — not production art.")
    ss = st.session_state
    fmt = st.radio("Output format", ["Image mockup", "HTML mockup"], horizontal=True,
                   help="HTML keeps real, selectable text and renders live; the image is a flat render.")
    if st.button("Generate improved design", icon=":material/auto_awesome:", disabled=not images):
        run_settings = Settings(openrouter_api_key=cfg["api_key"], default_model=cfg["model"])
        if fmt == "HTML mockup":
            with st.spinner("Coding an improved mockup as HTML…"):
                factory = LLMFactory(run_settings)
                ss.redesign = {"kind": "html",
                               "data": run_redesign_html(factory, cfg["model"], consolidated, images)}
        else:
            with st.spinner("Designing an improved mockup…"):
                ss.redesign = {"kind": "image",
                               "data": run_redesign(consolidated, images, run_settings)}
    if ss.get("redesign"):
        if ss.redesign.get("kind") == "html":
            render_redesign_html(ss.redesign["data"])
        else:
            render_redesign(ss.redesign["data"])


# --------------------------------------------------------------------- phases
def phase_setup(cfg: dict) -> None:
    st.title("Multimodal AI Design Analysis Suite")
    st.write("Upload or paste one or more screens. A LangGraph fleet critiques each through its "
             "own lens, grounds findings in design principles, marks the problem regions on the "
             "image, and a coordinator merges everything. Then ask follow-up questions.")
    images = collect_images()
    context = st.text_area("Optional context",
                           placeholder="e.g. 'SaaS pricing page for non-technical SMB owners.'", height=80)
    if st.button("Run critique", type="primary", disabled=not images):
        if not cfg["api_key"].strip():
            st.error("Enter your OpenRouter API key in the sidebar."); return
        if not cfg["agents"]:
            st.error("Select at least one agent."); return
        run_analysis(cfg, images, context)


def run_analysis(cfg: dict, images: list[dict], context: str) -> None:
    ss = st.session_state
    run_settings = Settings(openrouter_api_key=cfg["api_key"], default_model=cfg["model"],
                            retrieval_k=cfg["retrieval_k"], temperature=cfg["temperature"],
                            dedup_similarity_threshold=cfg["dedup"])
    kb = get_kb(run_settings.lancedb_path, run_settings.embedding_model)
    graph = build_graph(LLMFactory(run_settings), kb, run_settings)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # store only b64+mime in graph state (serializable for the checkpointer)
    state_images = [{"b64": im["b64"], "mime": im["mime"]} for im in images]
    initial = {"images": state_images, "model": cfg["model"], "selected_agents": cfg["agents"],
               "extra_context": context, "reports": [], "messages": []}

    selected = cfg["agents"]
    status = st.status("Analysing your screens…", expanded=True)
    # Pre-list every selected agent as "running" so you can see who's working,
    # not just who finished. The agents run in parallel, so they complete in
    # whatever order their model call returns — wall-clock is the slowest one.
    with status:
        agent_slots = {name: st.empty() for name in selected}
        for name, slot in agent_slots.items():
            slot.markdown(f":material/hourglass_top: {name} — running…")
        coord_slot = st.empty()
        coord_slot.markdown(":material/hourglass_empty: Coordinator — waiting for agents…")

    summary_box = st.empty()
    summary, done_agents = "", 0
    try:
        for mode, chunk in graph.stream(initial, config, stream_mode=["updates", "messages"]):
            if mode == "updates":
                for node, update in chunk.items():
                    if node == "agent":
                        done_agents += 1
                        report = (update.get("reports") or [{}])[0]
                        name = report.get("agent", "")
                        slot = agent_slots.get(name)
                        if slot is not None:
                            slot.markdown(f":material/check_circle: {name} — done")
                        # Once all agents are in, the coordinator (dedup + summary) runs.
                        if done_agents >= len(selected):
                            coord_slot.markdown(":material/hourglass_top: Coordinator — "
                                                "synthesising findings…")
                        else:
                            coord_slot.markdown(f":material/hourglass_empty: Coordinator — "
                                                f"waiting ({done_agents}/{len(selected)} agents done)")
                    elif node == "coordinator":
                        coord_slot.markdown(":material/check_circle: Coordinator — consolidated findings")
            elif mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "coordinator" and getattr(msg, "content", ""):
                    summary += msg.content
                    summary_box.markdown("**Executive summary (streaming):**\n\n" + summary + "▌")
        status.update(label="Analysis complete", state="complete", expanded=True)
    except Exception as exc:  # noqa: BLE001
        status.update(label="Analysis failed", state="error")
        st.exception(exc); return

    # keep names alongside the stored images for nicer annotated tabs
    ss.image_meta = [{"name": im["name"]} for im in images]
    ss.graph, ss.thread_id, ss.phase = graph, thread_id, "chat"
    st.rerun()


def phase_chat(cfg: dict) -> None:
    ss = st.session_state
    graph = ss.graph
    config = {"configurable": {"thread_id": ss.thread_id}}
    values = graph.get_state(config).values

    images = []
    names = ss.get("image_meta", [])
    for i, im in enumerate(values.get("images", [])):
        images.append({"b64": im["b64"], "mime": im["mime"],
                       "name": names[i]["name"] if i < len(names) else f"image-{i}"})

    st.title("Critique report")
    consolidated = values.get("consolidated")
    if consolidated:
        report_tab, redesign_tab = st.tabs([":material/description: Critique report",
                                            ":material/auto_awesome: Improved design"])
        with report_tab:
            render_report(ConsolidatedReport.model_validate(consolidated), images)
            render_cost_table(values.get("usage", []))
        with redesign_tab:
            render_redesign_tab(cfg, consolidated, images)

    st.divider()
    st.subheader("Ask about this design")
    for m in values.get("messages", []):
        role = "user" if m.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(m.content)

    q = st.chat_input("e.g. 'Which fix lifts conversion most?' or 'Explain finding 2.'")
    if q:
        with st.chat_message("user"):
            st.write(q)
        placeholder = st.chat_message("assistant").empty()
        answer = ""
        for mode, chunk in graph.stream(Command(resume=q), config, stream_mode=["messages"]):
            msg, meta = chunk
            if meta.get("langgraph_node") == "chat" and getattr(msg, "content", ""):
                answer += msg.content
                placeholder.markdown(answer + "▌")
        placeholder.markdown(answer)
        st.rerun()


def main() -> None:
    init_state()
    apply_theme(dark=st.session_state.get("dark_mode", False))
    cfg = sidebar()
    if st.session_state.phase == "setup":
        phase_setup(cfg)
    else:
        phase_chat(cfg)


if __name__ == "__main__":
    main()
