"""
Offline pricing tests (no network). Run:
    PYTHONPATH=. python tests/test_cost.py
"""

from __future__ import annotations

from config import _DEFAULT_PRICE, estimate_cost, price_for


def test_price_for_specific_before_general():
    # Regression for the ordering contract: the more-specific slug must win.
    assert price_for("google/gemini-2.5-flash-lite") != price_for("google/gemini-2.5-flash")
    # The image model resolves to its own (or a more specific) entry, not the default.
    assert price_for("google/gemini-2.5-flash-image") != _DEFAULT_PRICE


def test_price_for_unknown_falls_back():
    assert price_for("some/unknown-model-xyz") == _DEFAULT_PRICE
    assert price_for("") == _DEFAULT_PRICE


def test_estimate_cost_math():
    price_in, price_out = price_for("openai/gpt-4o-mini")
    cost = estimate_cost("openai/gpt-4o-mini", 1_000_000, 2_000_000)
    assert abs(cost - (price_in + 2 * price_out)) < 1e-9
    assert estimate_cost("openai/gpt-4o-mini", 0, 0) == 0.0


if __name__ == "__main__":
    test_price_for_specific_before_general()
    test_price_for_unknown_falls_back()
    test_estimate_cost_math()
    print("All cost tests passed.")
