"""Tests for the LLM fallback router (agents/fallback.py).

The `FallBack` class is the circuit-breaker between the agent graph and the
configured LLM providers (ollama/gpt/gemini/groq): it walks a caller-supplied
`fallback_order`, tries each router in turn, and only raises once every
router has failed. Meant to run as a CI stage that gates merges to
agents/fallback.py, agents/llm_models.py, and agents/helpers.py.

These are unit tests: `client_llm.get_model` (the only seam that touches real
network/API-key-backed providers) is monkeypatched with fake chat models, so
no real Ollama/OpenAI/Gemini/Groq call is ever made and no API keys are
required to run this file.
"""
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agents.llm.fallback import FallBack
import agents.llm.fallback as fallback_module


class DummySchema(BaseModel):
    answer: str


class _FakeChatModel:
    """Stand-in for a langchain chat model returned by client_llm.get_model."""

    def __init__(self, content=None, fail_message=None, structured_result=None, structured_fail_message=None):
        self._content = content
        self._fail_message = fail_message
        self._structured_result = structured_result
        self._structured_fail_message = structured_fail_message

    def invoke(self, message):
        if self._fail_message:
            raise RuntimeError(self._fail_message)
        return SimpleNamespace(content=self._content)

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredModel(self._structured_result, self._structured_fail_message)


class _FakeStructuredModel:
    def __init__(self, result, fail_message):
        self._result = result
        self._fail_message = fail_message

    def invoke(self, message):
        if self._fail_message:
            raise RuntimeError(self._fail_message)
        return self._result


def _patch_get_model(monkeypatch, models_by_router: dict):
    """Route client_llm.get_model(router, model_name) -> models_by_router[router]."""

    def fake_get_model(router, model_name, **kwargs):
        try:
            return models_by_router[router]
        except KeyError:
            raise AssertionError(f"unexpected router requested from get_model: {router}")

    monkeypatch.setattr(fallback_module.client_llm, "get_model", fake_get_model)


# ---------------------------------------------------------------------------
# __init__ / router registration
# ---------------------------------------------------------------------------

def test_init_only_registers_provided_models():
    fb = FallBack(llm_gpt="gpt-4o-mini")
    assert fb.llms == {"gpt": "gpt-4o-mini"}


def test_init_with_no_models_registers_none():
    fb = FallBack()
    assert fb.llms == {}


def test_init_registers_all_provided_models():
    fb = FallBack(llm_ollama="llama3", llm_gpt="gpt-4o-mini", llm_gemini="gemini-1.5", llm_groq="mixtral")
    assert fb.llms == {
        "ollama": "llama3",
        "gpt": "gpt-4o-mini",
        "gemini": "gemini-1.5",
        "groq": "mixtral",
    }


# ---------------------------------------------------------------------------
# invoke() -- regular generation
# ---------------------------------------------------------------------------

def test_invoke_returns_first_successful_router(monkeypatch):
    _patch_get_model(monkeypatch, {"gpt": _FakeChatModel(content="gpt answer")})
    fb = FallBack(llm_gpt="gpt-4o-mini")

    result = fb.invoke("what is 2+2?", fallback_order=["gpt"])

    assert result == "gpt answer"


def test_invoke_falls_back_to_next_router_on_failure(monkeypatch):
    _patch_get_model(monkeypatch, {
        "gpt": _FakeChatModel(fail_message="rate limited"),
        "gemini": _FakeChatModel(content="gemini answer"),
    })
    fb = FallBack(llm_gpt="gpt-4o-mini", llm_gemini="gemini-1.5")

    result = fb.invoke("question", fallback_order=["gpt", "gemini"])

    assert result == "gemini answer"


def test_invoke_stops_at_first_success_and_ignores_later_routers(monkeypatch):
    calls = []

    def fake_get_model(router, model_name):
        calls.append(router)
        return _FakeChatModel(content=f"{router} answer")

    monkeypatch.setattr(fallback_module.client_llm, "get_model", fake_get_model)
    fb = FallBack(llm_gpt="gpt-4o-mini", llm_gemini="gemini-1.5")

    result = fb.invoke("question", fallback_order=["gpt", "gemini"])

    assert result == "gpt answer"
    assert calls == ["gpt"]


def test_invoke_raises_runtime_error_when_all_routers_fail(monkeypatch):
    _patch_get_model(monkeypatch, {
        "gpt": _FakeChatModel(fail_message="gpt down"),
        "gemini": _FakeChatModel(fail_message="gemini down"),
    })
    fb = FallBack(llm_gpt="gpt-4o-mini", llm_gemini="gemini-1.5")

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        fb.invoke("question", fallback_order=["gpt", "gemini"])


def test_invoke_skips_router_with_no_configured_model(monkeypatch):
    _patch_get_model(monkeypatch, {"gemini": _FakeChatModel(content="gemini answer")})
    # "gpt" is listed in fallback_order but never configured on this instance.
    fb = FallBack(llm_gemini="gemini-1.5")

    result = fb.invoke("question", fallback_order=["gpt", "gemini"])

    assert result == "gemini answer"


def test_invoke_unsupported_router_name_is_skipped_not_raised(monkeypatch):
    _patch_get_model(monkeypatch, {"gpt": _FakeChatModel(content="gpt answer")})
    fb = FallBack(llm_gpt="gpt-4o-mini")

    result = fb.invoke("question", fallback_order=["not-a-real-router", "gpt"])

    assert result == "gpt answer"


def test_invoke_error_message_includes_each_router_failure(monkeypatch):
    _patch_get_model(monkeypatch, {
        "gpt": _FakeChatModel(fail_message="gpt-specific failure"),
        "gemini": _FakeChatModel(fail_message="gemini-specific failure"),
    })
    fb = FallBack(llm_gpt="gpt-4o-mini", llm_gemini="gemini-1.5")

    with pytest.raises(RuntimeError) as excinfo:
        fb.invoke("question", fallback_order=["gpt", "gemini"])

    assert "gpt-specific failure" in str(excinfo.value)
    assert "gemini-specific failure" in str(excinfo.value)


# ---------------------------------------------------------------------------
# constrained_invoke() -- structured/schema-bound generation
# ---------------------------------------------------------------------------

def test_constrained_invoke_requires_a_schema():
    fb = FallBack(llm_gpt="gpt-4o-mini")

    with pytest.raises(ValueError, match="constraine_model"):
        fb.constrained_invoke("question", fallback_order=["gpt"], constraine_model=None)


def test_constrained_invoke_returns_parsed_dict_on_success(monkeypatch):
    _patch_get_model(monkeypatch, {
        "gpt": _FakeChatModel(structured_result=DummySchema(answer="42")),
    })
    fb = FallBack(llm_gpt="gpt-4o-mini")

    result = fb.constrained_invoke("question", fallback_order=["gpt"], constraine_model=DummySchema)

    assert result == {"answer": "42"}


def test_constrained_invoke_falls_back_to_next_router_on_failure(monkeypatch):
    _patch_get_model(monkeypatch, {
        "gpt": _FakeChatModel(structured_fail_message="schema violation"),
        "groq": _FakeChatModel(structured_result=DummySchema(answer="fallback answer")),
    })
    fb = FallBack(llm_gpt="gpt-4o-mini", llm_groq="mixtral")

    result = fb.constrained_invoke("question", fallback_order=["gpt", "groq"], constraine_model=DummySchema)

    assert result == {"answer": "fallback answer"}


def test_constrained_invoke_raises_runtime_error_when_all_routers_fail(monkeypatch):
    _patch_get_model(monkeypatch, {
        "gpt": _FakeChatModel(structured_fail_message="gpt schema failure"),
        "groq": _FakeChatModel(structured_fail_message="groq schema failure"),
    })
    fb = FallBack(llm_gpt="gpt-4o-mini", llm_groq="mixtral")

    with pytest.raises(RuntimeError, match="constrained output"):
        fb.constrained_invoke("question", fallback_order=["gpt", "groq"], constraine_model=DummySchema)


class _ToolUseFailedError(Exception):
    """Mimics groq.BadRequestError: a tool_use_failed error whose body still
    carries the model's valid JSON answer under error.failed_generation."""

    def __init__(self, failed_generation: str):
        super().__init__("Tool choice is required, but model did not call a tool")
        self.body = {
            "error": {
                "message": "Tool choice is required, but model did not call a tool",
                "type": "invalid_request_error",
                "code": "tool_use_failed",
                "failed_generation": failed_generation,
            }
        }


def test_constrained_invoke_recovers_valid_json_from_tool_use_failed_error(monkeypatch):
    fake_groq = _FakeChatModel()
    fake_groq.with_structured_output = lambda schema: _RaisingStructuredModel(
        _ToolUseFailedError('{"answer": "42"}')
    )
    _patch_get_model(monkeypatch, {"groq": fake_groq})
    fb = FallBack(llm_groq="mixtral")

    result = fb.constrained_invoke("question", fallback_order=["groq"], constraine_model=DummySchema)

    assert result == {"answer": "42"}


def test_constrained_invoke_falls_back_when_failed_generation_is_invalid(monkeypatch):
    fake_groq = _FakeChatModel()
    fake_groq.with_structured_output = lambda schema: _RaisingStructuredModel(
        _ToolUseFailedError('{"not_a_real_field": true}')
    )
    _patch_get_model(monkeypatch, {
        "groq": fake_groq,
        "gpt": _FakeChatModel(structured_result=DummySchema(answer="gpt saved it")),
    })
    fb = FallBack(llm_groq="mixtral", llm_gpt="gpt-4o-mini")

    result = fb.constrained_invoke("question", fallback_order=["groq", "gpt"], constraine_model=DummySchema)

    assert result == {"answer": "gpt saved it"}


class _RaisingStructuredModel:
    def __init__(self, error: Exception):
        self._error = error

    def invoke(self, message):
        raise self._error


def test_constrained_invoke_forwards_method_to_with_structured_output(monkeypatch):
    captured_kwargs = {}

    class _RecordingChatModel:
        def with_structured_output(self, schema, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeStructuredModel(DummySchema(answer="42"), None)

    _patch_get_model(monkeypatch, {"groq": _RecordingChatModel()})
    fb = FallBack(llm_groq="mixtral")

    fb.constrained_invoke("question", fallback_order=["groq"], constraine_model=DummySchema, method="json_schema")

    assert captured_kwargs == {"method": "json_schema"}


def test_constrained_invoke_omits_method_by_default(monkeypatch):
    captured_kwargs = {}

    class _RecordingChatModel:
        def with_structured_output(self, schema, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeStructuredModel(DummySchema(answer="42"), None)

    _patch_get_model(monkeypatch, {"groq": _RecordingChatModel()})
    fb = FallBack(llm_groq="mixtral")

    fb.constrained_invoke("question", fallback_order=["groq"], constraine_model=DummySchema)

    assert captured_kwargs == {}


def test_constrained_invoke_forwards_reasoning_effort_to_groq_get_model(monkeypatch):
    captured = {}

    def fake_get_model(router, model_name, **kwargs):
        captured[router] = kwargs
        return _FakeChatModel(structured_result=DummySchema(answer="42"))

    monkeypatch.setattr(fallback_module.client_llm, "get_model", fake_get_model)
    fb = FallBack(llm_groq="mixtral")

    fb.constrained_invoke(
        "question", fallback_order=["groq"], constraine_model=DummySchema, groq_reasoning_effort="low"
    )

    assert captured == {"groq": {"reasoning_effort": "low"}}


def test_constrained_invoke_does_not_forward_reasoning_effort_to_other_routers(monkeypatch):
    captured = {}

    def fake_get_model(router, model_name, **kwargs):
        captured[router] = kwargs
        return _FakeChatModel(structured_result=DummySchema(answer="42"))

    monkeypatch.setattr(fallback_module.client_llm, "get_model", fake_get_model)
    fb = FallBack(llm_gpt="gpt-4o-mini")

    fb.constrained_invoke(
        "question", fallback_order=["gpt"], constraine_model=DummySchema, groq_reasoning_effort="low"
    )

    assert captured == {"gpt": {}}
