"""
Annotation rendering.

Turns a screenshot plus the findings that point at it into a marked-up image:
- each locatable finding gets a severity-coloured box and a numbered pin,
- a legend panel listing "N. [severity] title — recommendation" is baked into
  the image so a single downloadable PNG carries both the marks and the comments.

Pure PIL, no Streamlit — so it can be unit-tested headless.
"""

from __future__ import annotations

import io
import textwrap

from PIL import Image, ImageDraw, ImageFont

SEVERITY_RGB = {
    "critical": (209, 17, 73),
    "high": (241, 113, 5),
    "medium": (214, 160, 30),
    "low": (58, 134, 255),
}
NEUTRAL = (110, 110, 110)
LEGEND_W = 360
PAD = 16


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_pin(draw: ImageDraw.ImageDraw, x: float, y: float, number: int, color, font) -> None:
    r = 14
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color + (255,), outline=(255, 255, 255, 255), width=2)
    label = str(number)
    tw, th = _text_size(draw, label, font)
    draw.text((x - tw / 2, y - th / 2 - 1), label, fill=(255, 255, 255, 255), font=font)


def annotate_image(image_bytes: bytes, items: list[dict], with_legend: bool = True) -> bytes:
    """Return a PNG with boxes + numbered pins (and an optional comments legend).

    ``items``: list of dicts with keys
        number (int), severity (str), title (str), recommendation (str),
        region (dict {x,y,w,h} normalized) | None
    """
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pin_font = _font(max(13, W // 70))

    for it in items:
        color = SEVERITY_RGB.get(it.get("severity", ""), NEUTRAL)
        region = it.get("region")
        if not region:
            continue
        x, y = region["x"] * W, region["y"] * H
        w, h = region["w"] * W, region["h"] * H
        draw.rectangle([x, y, x + w, y + h], outline=color + (255,), width=3)
        draw.rectangle([x, y, x + w, y + h], fill=color + (40,))  # faint tint
        _draw_pin(draw, x, y, it["number"], color, pin_font)

    annotated = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    if not with_legend or not items:
        return _to_png(annotated)
    return _to_png(_attach_legend(annotated, items))


def _attach_legend(annotated: Image.Image, items: list[dict]) -> Image.Image:
    W, H = annotated.size
    title_font = _font(15)
    body_font = _font(13)
    badge_font = _font(12)

    # Measure the legend height first.
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    line_h = _text_size(probe, "Ag", body_font)[1] + 4
    rows: list[tuple[dict, list[str]]] = []
    total = PAD + line_h + 6
    for it in items:
        wrapped = textwrap.wrap(
            f"[{it.get('severity', '')}] {it['title']} — {it['recommendation']}",
            width=46,
        ) or [it["title"]]
        rows.append((it, wrapped))
        total += line_h * (len(wrapped)) + 12
    legend_h = max(H, total + PAD)

    canvas = Image.new("RGB", (W + LEGEND_W, legend_h), (255, 255, 255))
    canvas.paste(annotated, (0, 0))
    d = ImageDraw.Draw(canvas)
    x0 = W + PAD
    d.text((x0, PAD), "Findings", fill=(20, 20, 20), font=title_font)
    cy = PAD + line_h + 6
    for it, wrapped in rows:
        color = SEVERITY_RGB.get(it.get("severity", ""), NEUTRAL)
        d.ellipse([x0, cy, x0 + 22, cy + 22], fill=color)
        n = str(it["number"])
        tw, th = _text_size(d, n, badge_font)
        d.text((x0 + 11 - tw / 2, cy + 11 - th / 2 - 1), n, fill=(255, 255, 255), font=badge_font)
        ty = cy
        for line in wrapped:
            d.text((x0 + 30, ty), line, fill=(40, 40, 40), font=body_font)
            ty += line_h
        cy = ty + 12
    return canvas


def _to_png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
