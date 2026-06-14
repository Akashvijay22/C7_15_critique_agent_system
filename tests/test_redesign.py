"""
Offline redesign-agent tests (no network — generate_image is faked). Run:
    PYTHONPATH=. python tests/test_redesign.py
"""

from __future__ import annotations

from config import Settings
from src.agents import redesign


def _report():
    """A consolidated report with more findings than the prompt caps allow."""
    findings = [{"title": f"Issue {i}", "recommendation": f"Fix {i}"} for i in range(20)]
    agent_reports = [
        {"agent": "Visual Analysis Agent", "error": None,
         "findings": [{"title": f"V{i}", "recommendation": f"vis fix {i}"} for i in range(5)]},
        {"agent": "UX Critique Agent", "error": None,
         "findings": [{"title": f"U{i}", "recommendation": f"ux fix {i}"} for i in range(5)]},
        {"agent": "Broken Agent", "error": "boom", "findings": []},
    ]
    return {"prioritised_findings": findings, "agent_reports": agent_reports,
            "executive_summary": "Overall, raise contrast."}


def test_prompt_is_bounded():
    prompt = redesign.build_redesign_prompt(_report())
    # ≤ 8 prioritised fixes
    assert sum(1 for ln in prompt.splitlines() if ln.startswith("- Fix ")) == redesign.MAX_PRIORITISED
    # ≤ 6 specialist recs total, ≤ 2 per agent
    rec_lines = [ln for ln in prompt.splitlines() if ln.startswith("- (")]
    assert len(rec_lines) == redesign.MAX_RECS_TOTAL
    assert sum(1 for ln in rec_lines if ln.startswith("- (Visual")) <= redesign.MAX_RECS_PER_AGENT
    assert "Constraints:" in prompt


def test_run_redesign_success(monkeypatch):
    def fake_generate_image(settings, model, prompt, images, **kw):
        assert len(images) <= 1            # primary design only
        return [b"PNGBYTES"], "here you go", {"cost": 0.012}

    monkeypatch.setattr(redesign.llm, "generate_image", fake_generate_image)
    out = redesign.run_redesign(_report(), [{"b64": "eHg=", "mime": "image/png"}],
                                Settings(openrouter_api_key="x"))
    assert out["images"] == [b"PNGBYTES"]
    assert out["cost"] == 0.012           # OpenRouter-exact cost preferred
    assert out["error"] is None
    assert out["latency"] >= 0            # A6: latency is reported


def test_run_redesign_estimates_cost_when_missing(monkeypatch):
    def fake_generate_image(settings, model, prompt, images, **kw):
        return [b"x"], "", {"prompt_tokens": 1000, "completion_tokens": 0}

    monkeypatch.setattr(redesign.llm, "generate_image", fake_generate_image)
    out = redesign.run_redesign(_report(), [{"b64": "eHg=", "mime": "image/png"}],
                                Settings(openrouter_api_key="x"))
    assert out["cost"] > 0                # fell back to a token estimate
    assert out["error"] is None


class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 100, "output_tokens": 500, "total_tokens": 600}


class _FakeChat:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _FakeMsg(self._content)


class _FakeFactory:
    def __init__(self, content):
        self._content = content

    def chat(self, model, **kwargs):
        return _FakeChat(self._content)


def test_run_redesign_html_strips_fences_and_costs():
    fenced = "```html\n<!doctype html><html><body>Hi</body></html>\n```"
    out = redesign.run_redesign_html(_FakeFactory(fenced), "openai/gpt-4o-mini",
                                     _report(), [{"b64": "eHg=", "mime": "image/png"}])
    assert out["html"].startswith("<!doctype html>")
    assert "```" not in out["html"]
    assert out["cost"] > 0 and out["error"] is None and out["latency"] >= 0


def test_run_redesign_html_never_raises():
    class Boom:
        def chat(self, model, **kwargs):
            raise RuntimeError("500: server error")

    out = redesign.run_redesign_html(Boom(), "m", _report(),
                                     [{"b64": "eHg=", "mime": "image/png"}])
    assert out["html"] == "" and "500" in out["error"]


def test_run_redesign_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("429: rate limited")

    monkeypatch.setattr(redesign.llm, "generate_image", boom)
    out = redesign.run_redesign(_report(), [{"b64": "eHg=", "mime": "image/png"}],
                                Settings(openrouter_api_key="x"))
    assert out["images"] == []
    assert "429" in out["error"]
    assert "latency" in out


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
