"""
LangChain model layer.

``LLMFactory`` builds ``ChatOpenAI`` instances pointed at OpenRouter (which is
OpenAI-API compatible). Centralising construction here means the rest of the app
asks for "a chat model for this model id" and never touches base URLs, headers,
or keys.

Why ChatOpenAI rather than a bespoke client:
- ``with_structured_output(Schema)`` gives us validated Pydantic output for free
  — this is the "fixed response format" guarantee, enforced by the framework.
- Streaming, retries, callbacks, and LangSmith tracing come built in.
- LangGraph's ``stream_mode="messages"`` taps the model's token stream
  automatically, so summary/chat tokens surface live without extra plumbing.

Multimodal input uses LangChain message content blocks; the OpenAI-style
``image_url`` block with a base64 data URL is what OpenRouter expects.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import Settings

T = TypeVar("T", bound=BaseModel)


def data_url(b64: str, mime: str) -> str:
    return f"data:{mime};base64,{b64}"


def usage_from_message(message) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` from a LangChain AIMessage.

    Returns ``(0, 0)`` when no usage metadata is present (e.g. mock/fake runs),
    so callers never have to special-case missing usage.
    """
    meta = getattr(message, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


class LLMFactory:
    """Creates configured chat models. One instance is shared across the graph."""

    def __init__(self, settings: Settings):
        if not settings.has_api_key:
            raise ValueError(
                "No OPENROUTER_API_KEY configured. Set it in the environment or the sidebar."
            )
        self._settings = settings
        # OpenRouter uses these headers for app attribution / rankings.
        self._default_headers = {
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        }

    def chat(self, model: str, *, streaming: bool = True, temperature: float | None = None,
             max_tokens: int | None = None) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            base_url=self._settings.base_url,
            api_key=self._settings.openrouter_api_key,
            temperature=self._settings.temperature if temperature is None else temperature,
            max_tokens=self._settings.max_tokens if max_tokens is None else max_tokens,
            timeout=self._settings.request_timeout,
            streaming=streaming,
            default_headers=self._default_headers,
        )

    def structured(self, model: str, schema: Type[T], *, include_raw: bool = False):
        """A runnable that returns a validated ``schema`` instance.

        With ``include_raw=True`` the runnable instead returns a dict
        ``{"raw": AIMessage, "parsed": schema, "parsing_error": ...}`` so callers
        can read token usage off the raw message for the cost table.
        """
        return self.chat(model, streaming=False).with_structured_output(
            schema, method=self._settings.structured_output_method, include_raw=include_raw
        )


def multimodal_messages(
    system: str,
    user_text: str,
    images: list[dict] | None = None,
) -> list[BaseMessage]:
    """Build a [System, Human] message pair with optional images.

    ``images`` is a list of ``{"b64": str, "mime": str}``. Text precedes images,
    per OpenRouter's guidance for best vision results.
    """
    content: list[dict] = [{"type": "text", "text": user_text}]
    for img in images or []:
        content.append(
            {"type": "image_url", "image_url": {"url": data_url(img["b64"], img["mime"])}}
        )
    return [SystemMessage(content=system), HumanMessage(content=content)]


def generate_image(
    settings: Settings,
    model: str,
    prompt: str,
    images: list[dict] | None = None,
    *,
    timeout: int = 180,
) -> tuple[list[bytes], str, dict]:
    """Generate image(s) from a prompt + reference images via OpenRouter.

    Image generation does **not** go through ``ChatOpenAI`` — LangChain does not
    surface images that the model *returns*. So this calls OpenRouter's
    ``/chat/completions`` directly with ``urllib`` and asks for image output.

    ``images`` is a list of ``{"b64": str, "mime": str}`` reference designs,
    inlined as base64 ``data:`` URLs.

    Returns ``(images, text, usage)``:
    - ``images``: decoded PNG/JPEG bytes read from ``message.images[].image_url.url``
    - ``text``:   any prose the model returned alongside the image
    - ``usage``:  OpenRouter's ``usage`` object (carries an exact ``cost`` for image gen)

    Raises ``RuntimeError("<code>: <body excerpt>")`` on HTTP errors; the UI catches it.
    """
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images or []:
        content.append(
            {"type": "image_url", "image_url": {"url": data_url(img["b64"], img["mime"])}}
        )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
    }
    req = urllib.request.Request(
        f"{settings.base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # surface code + body so the UI can show it
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{exc.code}: {detail}") from exc

    message = (payload.get("choices") or [{}])[0].get("message", {}) or {}
    out_images: list[bytes] = []
    for item in message.get("images") or []:
        url = (item.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and "," in url:
            try:
                out_images.append(base64.b64decode(url.split(",", 1)[1]))
            except Exception:  # skip a malformed data URL rather than crash
                pass
    raw_content = message.get("content")
    text = raw_content if isinstance(raw_content, str) else ""
    usage = payload.get("usage") or {}
    return out_images, text, usage
