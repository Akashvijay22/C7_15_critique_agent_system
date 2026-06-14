"""
Annotation renderer test (headless PIL, no network). Run:
    PYTHONPATH=. python tests/test_annotate.py
"""

from __future__ import annotations

import io

from PIL import Image

from src.core.annotate import annotate_image


def _blank(w=400, h=300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (245, 245, 245)).save(buf, "PNG")
    return buf.getvalue()


def test_boxes_and_legend():
    img = _blank()
    items = [
        {"number": 1, "severity": "critical", "title": "Low contrast CTA",
         "recommendation": "Raise contrast to 4.5:1", "region": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.12}},
        {"number": 2, "severity": "medium", "title": "Cluttered nav",
         "recommendation": "Reduce nav items", "region": None},  # not localised
    ]
    plain = annotate_image(img, items, with_legend=False)
    legended = annotate_image(img, items, with_legend=True)

    a = Image.open(io.BytesIO(plain))
    b = Image.open(io.BytesIO(legended))
    assert a.size == (400, 300)
    assert b.size[0] > 400          # legend panel widened the canvas
    assert b.format == "PNG"


def test_no_items_returns_original_size():
    img = _blank()
    out = annotate_image(img, [], with_legend=True)
    assert Image.open(io.BytesIO(out)).size == (400, 300)


if __name__ == "__main__":
    test_boxes_and_legend()
    test_no_items_returns_original_size()
    print("Annotation tests passed.")
