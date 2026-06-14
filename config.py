"""
Central configuration and the model registry.

Everything tunable lives here so the rest of the codebase never hard-codes a
URL, a model id, or a magic number. Reads from environment variables (and a
local ``.env`` if present) with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: load a local .env during development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model: the user-requested GPT-4o-mini (cheap, fast, vision-capable).
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Default image-output model for the on-demand Redesign agent. Must be a model
# that can *return* an image (not just read one). Override with env REDESIGN_MODEL.
DEFAULT_REDESIGN_MODEL = "google/gemini-2.5-flash-image"


@dataclass(frozen=True)
class ModelSpec:
    """A selectable model exposed in the UI dropdown."""

    id: str               # OpenRouter model id
    label: str            # human-friendly name
    vision: bool          # can it read images?
    notes: str = ""


# Curated short-list of vision-capable models available on OpenRouter.
# Vision is required because the agents reason over screenshots.
SUPPORTED_MODELS: list[ModelSpec] = [
    ModelSpec("openai/gpt-4o-mini", "GPT-4o mini (default)", True,
              "Cheapest reliable vision model. Great default."),
    ModelSpec("openai/gpt-4o", "GPT-4o", True,
              "Stronger reasoning, higher cost."),
    ModelSpec("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet", True,
              "Excellent at structured critique and nuance."),
    ModelSpec("google/gemini-flash-1.5", "Gemini 1.5 Flash", True,
              "Fast and inexpensive, large context."),
    ModelSpec("qwen/qwen2.5-vl-72b-instruct", "Qwen2.5-VL 72B", True,
              "Strong open-weight vision model."),
]

MODELS_BY_ID = {m.id: m for m in SUPPORTED_MODELS}


@dataclass
class Settings:
    """Runtime settings, resolved once at startup."""

    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = OPENROUTER_BASE_URL
    default_model: str = field(default_factory=lambda: os.getenv("CRITIQUE_MODEL", DEFAULT_MODEL))
    redesign_model: str = field(default_factory=lambda: os.getenv("REDESIGN_MODEL", DEFAULT_REDESIGN_MODEL))

    # Identify the app to OpenRouter (used for rankings / abuse handling).
    app_url: str = field(default_factory=lambda: os.getenv("APP_URL", "https://localhost"))
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "Multimodal UI/UX Critique Suite"))

    # RAG / vector store
    lancedb_path: str = field(default_factory=lambda: os.getenv("LANCEDB_PATH", "./.lancedb"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    kb_table: str = "design_knowledge"
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    # LLM behaviour
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    request_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))

    # Structured output method used by LangChain `with_structured_output`.
    # "function_calling" works for tool-capable models (the default fleet);
    # switch to "json_mode" for models without tool support.
    structured_output_method: str = field(
        default_factory=lambda: os.getenv("STRUCTURED_OUTPUT_METHOD", "function_calling")
    )

    # Coordinator
    dedup_similarity_threshold: float = float(os.getenv("DEDUP_THRESHOLD", "0.82"))

    @property
    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key.strip())


settings = Settings()


# --------------------------------------------------------------------- pricing
# Approximate OpenRouter list prices in USD per **1M tokens** as ``(input, output)``.
#
# This is a *display aid* for the per-run cost table, not billing. It is
# hand-maintained — adjust it to track OpenRouter's live rates.
#
# ``price_for`` returns the FIRST substring match, so **order matters**: more
# specific slugs must precede the general ones (e.g. ``gemini-2.5-flash-lite``
# and ``gemini-2.5-flash-image`` before ``gemini-2.5-flash``).
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Anthropic
    "claude-3.5-sonnet": (3.00, 15.00),
    # Google — specific before general
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash-image": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-flash-1.5": (0.075, 0.30),
    # Open weights
    "qwen2.5-vl-72b": (0.40, 0.40),
}

# Fallback when a model id matches none of the keys above.
_DEFAULT_PRICE: tuple[float, float] = (1.0, 3.0)


def price_for(model: str) -> tuple[float, float]:
    """Return ``(input, output)`` price per 1M tokens for ``model``.

    Scans ``PRICING`` in order and returns the first substring match, so the
    most-specific keys must come first. Unknown models get ``_DEFAULT_PRICE``.
    """
    slug = (model or "").lower()
    for key, price in PRICING.items():
        if key in slug:
            return price
    return _DEFAULT_PRICE


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a call from its token counts.

    ``cost = prompt_tokens/1e6 * price_in + completion_tokens/1e6 * price_out``.
    Approximate by design — used for the cost table, never for billing.
    """
    price_in, price_out = price_for(model)
    return (prompt_tokens / 1e6) * price_in + (completion_tokens / 1e6) * price_out
