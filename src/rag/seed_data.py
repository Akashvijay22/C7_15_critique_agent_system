"""
Seed corpus for the design knowledge base.

Each entry is a small, self-contained principle with a real source. These are
embedded into LanceDB at startup; agents retrieve the most relevant entries for
a given screen and cite them in their findings. That is what makes the tool's
output *grounded* rather than free-floating opinion.

Keep entries short and atomic — one principle each — so retrieval stays sharp.
"""

from __future__ import annotations

from typing import TypedDict


class KBEntry(TypedDict):
    kb_id: str
    category: str          # one of: visual | ux | accessibility | market | content
    title: str
    text: str
    source: str
    url: str


SEED_KNOWLEDGE: list[KBEntry] = [
    # ---- Nielsen's 10 usability heuristics (UX) ----
    {
        "kb_id": "nn-h1", "category": "ux",
        "title": "Visibility of system status",
        "text": "The design should keep users informed about what is going on through "
                "appropriate, timely feedback. Loading states, progress, and confirmations "
                "reduce uncertainty.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h2", "category": "ux",
        "title": "Match between system and the real world",
        "text": "Use words, phrases, and concepts familiar to the user rather than internal "
                "jargon. Follow real-world conventions and a natural, logical order.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h3", "category": "ux",
        "title": "User control and freedom",
        "text": "Users need a clearly marked emergency exit — undo and redo — to leave "
                "unwanted states without an extended dialogue.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h4", "category": "ux",
        "title": "Consistency and standards",
        "text": "Users should not have to wonder whether different words, situations, or "
                "actions mean the same thing. Follow platform and industry conventions.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h5", "category": "ux",
        "title": "Error prevention",
        "text": "Prevent problems before they occur. Eliminate error-prone conditions or "
                "present a confirmation before users commit to an action.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h6", "category": "ux",
        "title": "Recognition rather than recall",
        "text": "Minimise memory load by making elements, actions, and options visible. "
                "Users should not have to remember information across the interface.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h8", "category": "ux",
        "title": "Aesthetic and minimalist design",
        "text": "Interfaces should not contain information that is irrelevant or rarely "
                "needed. Every extra unit of content competes with the relevant units.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },
    {
        "kb_id": "nn-h9", "category": "ux",
        "title": "Help users recognise, diagnose, and recover from errors",
        "text": "Error messages should be in plain language, precisely indicate the problem, "
                "and constructively suggest a solution.",
        "source": "Nielsen Norman Group — 10 Usability Heuristics",
        "url": "https://www.nngroup.com/articles/ten-usability-heuristics/",
    },

    # ---- UX / cognitive laws ----
    {
        "kb_id": "law-hicks", "category": "ux",
        "title": "Hick's Law",
        "text": "The time to make a decision increases with the number and complexity of "
                "choices. Reduce options or chunk them to speed up decisions.",
        "source": "Laws of UX — Hick's Law",
        "url": "https://lawsofux.com/hicks-law/",
    },
    {
        "kb_id": "law-fitts", "category": "ux",
        "title": "Fitts's Law",
        "text": "The time to acquire a target is a function of the distance to and size of "
                "the target. Make primary actions large and place them within easy reach.",
        "source": "Laws of UX — Fitts's Law",
        "url": "https://lawsofux.com/fittss-law/",
    },
    {
        "kb_id": "law-zeigarnik", "category": "ux",
        "title": "Zeigarnik Effect",
        "text": "People remember uncompleted or interrupted tasks better than completed ones. "
                "Progress indicators encourage users to finish multi-step flows.",
        "source": "Laws of UX — Zeigarnik Effect",
        "url": "https://lawsofux.com/zeigarnik-effect/",
    },
    {
        "kb_id": "law-jakob", "category": "ux",
        "title": "Jakob's Law",
        "text": "Users spend most of their time on other sites, so they prefer your site to "
                "work the same way as the ones they already know.",
        "source": "Laws of UX — Jakob's Law",
        "url": "https://lawsofux.com/jakobs-law/",
    },
    {
        "kb_id": "law-miller", "category": "ux",
        "title": "Miller's Law",
        "text": "The average person can hold about 7 (±2) items in working memory. Chunk "
                "content into smaller groups to ease processing.",
        "source": "Laws of UX — Miller's Law",
        "url": "https://lawsofux.com/millers-law/",
    },

    # ---- Visual / typographic design ----
    {
        "kb_id": "vis-hierarchy", "category": "visual",
        "title": "Visual hierarchy",
        "text": "Use size, weight, colour, and spacing to signal importance so the eye is "
                "guided to the most important element first. One clear focal point per view.",
        "source": "Interaction Design Foundation — Visual Hierarchy",
        "url": "https://www.interaction-design.org/literature/topics/visual-hierarchy",
    },
    {
        "kb_id": "vis-whitespace", "category": "visual",
        "title": "Whitespace and proximity",
        "text": "Related elements should be grouped with consistent spacing; generous "
                "whitespace improves comprehension and perceived quality (Gestalt proximity).",
        "source": "Gestalt principles of grouping",
        "url": "https://www.nngroup.com/articles/gestalt-proximity/",
    },
    {
        "kb_id": "vis-typescale", "category": "visual",
        "title": "Typographic scale and line length",
        "text": "Limit the type system to a small modular scale; aim for 45-75 characters per "
                "line for body copy and at least 16px base size for readability.",
        "source": "Butterick's Practical Typography",
        "url": "https://practicaltypography.com/",
    },
    {
        "kb_id": "vis-contrast", "category": "visual",
        "title": "Colour and contrast discipline",
        "text": "Limit the palette, use colour consistently to encode meaning, and never rely "
                "on colour alone to convey information.",
        "source": "Material Design — Color system",
        "url": "https://m3.material.io/styles/color/system/overview",
    },

    # ---- Accessibility (WCAG) ----
    {
        "kb_id": "wcag-143", "category": "accessibility",
        "title": "WCAG 1.4.3 Contrast (Minimum) — AA",
        "text": "Text and images of text must have a contrast ratio of at least 4.5:1 (3:1 "
                "for large text of 18pt/14pt bold).",
        "source": "WCAG 2.2 — Success Criterion 1.4.3",
        "url": "https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html",
    },
    {
        "kb_id": "wcag-146", "category": "accessibility",
        "title": "WCAG 1.4.6 Contrast (Enhanced) — AAA",
        "text": "For AAA, text contrast should be at least 7:1 (4.5:1 for large text).",
        "source": "WCAG 2.2 — Success Criterion 1.4.6",
        "url": "https://www.w3.org/WAI/WCAG22/Understanding/contrast-enhanced.html",
    },
    {
        "kb_id": "wcag-253", "category": "accessibility",
        "title": "WCAG 2.5.3 Label in Name",
        "text": "For controls with visible labels, the accessible name must contain the "
                "visible text so speech-input users can operate them.",
        "source": "WCAG 2.2 — Success Criterion 2.5.3",
        "url": "https://www.w3.org/WAI/WCAG22/Understanding/label-in-name.html",
    },
    {
        "kb_id": "wcag-255", "category": "accessibility",
        "title": "WCAG 2.5.5 Target Size",
        "text": "Interactive targets should be at least 44x44 CSS pixels to be comfortably "
                "tappable, reducing mis-taps on touch devices.",
        "source": "WCAG 2.2 — Target Size",
        "url": "https://www.w3.org/WAI/WCAG22/Understanding/target-size-enhanced.html",
    },

    # ---- Market / conversion ----
    {
        "kb_id": "mkt-valueprop", "category": "market",
        "title": "Value proposition clarity",
        "text": "Within five seconds a visitor should understand what the product does, who "
                "it is for, and why it is better. Lead with outcomes, not features.",
        "source": "Marketing Examples / CXL — Value proposition",
        "url": "https://cxl.com/blog/value-proposition-examples/",
    },
    {
        "kb_id": "mkt-cta", "category": "market",
        "title": "Single primary call-to-action",
        "text": "Each view should have one dominant, action-oriented CTA. Competing CTAs of "
                "equal weight dilute conversion and create decision paralysis.",
        "source": "Nielsen Norman Group — Call to action buttons",
        "url": "https://www.nngroup.com/articles/clickable-elements/",
    },
    {
        "kb_id": "mkt-trust", "category": "market",
        "title": "Social proof and trust signals",
        "text": "Testimonials, logos, ratings, and security cues reduce perceived risk near "
                "the point of conversion.",
        "source": "CXL — Social proof",
        "url": "https://cxl.com/blog/social-proof/",
    },
    {
        "kb_id": "mkt-friction", "category": "market",
        "title": "Reduce conversion friction",
        "text": "Remove unnecessary form fields, defer account creation, and minimise steps "
                "between intent and completion to lift conversion.",
        "source": "Baymard Institute — Checkout usability",
        "url": "https://baymard.com/blog/checkout-flow-average-form-fields",
    },

    # ---- Content / UX writing (microcopy) ----
    {
        "kb_id": "cnt-clarity", "category": "content",
        "title": "Clarity over cleverness",
        "text": "UX writing should be clear, concise, and useful. Prefer plain, specific "
                "words over clever or generic phrasing; cut words that do not help the user "
                "act or decide.",
        "source": "Nielsen Norman Group — UX writing",
        "url": "https://www.nngroup.com/articles/ux-writing-study-guide/",
    },
    {
        "kb_id": "cnt-buttons", "category": "content",
        "title": "Action-oriented, specific button labels",
        "text": "Buttons should describe the action and its outcome (e.g. 'Start free trial') "
                "rather than vague labels like 'Submit' or 'OK', so users know what happens "
                "before they click.",
        "source": "Nielsen Norman Group — Better link & button labels",
        "url": "https://www.nngroup.com/articles/better-link-labels/",
    },
    {
        "kb_id": "cnt-errors", "category": "content",
        "title": "Helpful error messages",
        "text": "Error text should say what went wrong, why, and how to fix it, in plain "
                "human language without codes or blame. Place it next to the field it refers to.",
        "source": "Nielsen Norman Group — Error message guidelines",
        "url": "https://www.nngroup.com/articles/error-message-guidelines/",
    },
    {
        "kb_id": "cnt-readability", "category": "content",
        "title": "Readable, scannable copy",
        "text": "Write for a broad audience at a low reading grade level; use short sentences, "
                "front-loaded key points, and chunked text so users can scan rather than read "
                "word by word.",
        "source": "Nielsen Norman Group — Legibility, readability & comprehension",
        "url": "https://www.nngroup.com/articles/legibility-readability-comprehension/",
    },
]
