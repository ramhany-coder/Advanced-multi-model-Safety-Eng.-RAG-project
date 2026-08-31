"""Tests for the query-decomposition agent (agents/QueryDecomposer/agent.py).

This agent expands `state['merged']` into several corpus-worded retrieval
phrases so the retriever can search for each OSHA duty separately. Its
correctness has two independent halves:

1. The *contract* -- always keeps `merged` at position 0, never returns an
   empty list even on total LLM failure, dedupes/caps what the model
   proposes. That's what's covered here, as pure unit tests.
2. The *quality* of the decomposition itself (does it actually name the right
   duties in corpus vocabulary) -- that's a prompt-engineering property,
   exercised via the citation-recall golden set in tests/test_retrieval.py
   using a fixed, known-good sub-query list rather than a live, non-
   deterministic LLM call.

`client_llm.get_model` (the only seam that touches a real network/API-key-
backed provider) is monkeypatched with a fake chat model, same pattern as
tests/test_fallback.py -- so no real Groq/OpenAI call is ever made here and
no API keys are required to run this file.
"""
from types import SimpleNamespace

import agents.llm.fallback as fallback_module
from agents.QueryDecomposer.agent import _normalize_sub_queries, query_decomposer_agent
from agents.QueryDecomposer.schemas import QueryDecomposition


class _FakeStructuredModel:
    def __init__(self, result=None, fail_message=None):
        self._result = result
        self._fail_message = fail_message

    def invoke(self, message):
        if self._fail_message:
            raise RuntimeError(self._fail_message)
        return self._result


class _FakeChatModel:
    def __init__(self, result=None, fail_message=None):
        self._result = result
        self._fail_message = fail_message

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredModel(self._result, self._fail_message)


def _patch_get_model(monkeypatch, model):
    def fake_get_model(router, model_name, **kwargs):
        return model

    monkeypatch.setattr(fallback_module.client_llm, "get_model", fake_get_model)


# ---------------------------------------------------------------------------
# _normalize_sub_queries -- merged-first, dedupe, cap
# ---------------------------------------------------------------------------

def test_normalize_forces_merged_into_position_zero_even_if_model_omits_it():
    result = _normalize_sub_queries(["dump body tailgate trip handle"], "unload sand truck")
    assert result[0] == "unload sand truck"


def test_normalize_forces_merged_first_even_if_model_reorders_it():
    merged = "unload sand truck"
    result = _normalize_sub_queries([merged, "dump body tailgate trip handle", merged], merged)
    assert result[0] == merged
    assert result.count(merged) == 1  # deduped, not just re-inserted


def test_normalize_drops_duplicate_and_empty_entries():
    result = _normalize_sub_queries(
        ["duty one", "duty one", "  ", "", "duty two", "DUTY ONE"], "merged query"
    )
    assert result == ["merged query", "duty one", "duty two"]


def test_normalize_caps_at_six():
    raw = [f"duty {i}" for i in range(10)]
    result = _normalize_sub_queries(raw, "merged query")
    assert len(result) == 6
    assert result[0] == "merged query"


def test_normalize_ignores_non_string_entries():
    result = _normalize_sub_queries(["duty one", 42, None, ["nested"]], "merged query")
    assert result == ["merged query", "duty one"]


# ---------------------------------------------------------------------------
# query_decomposer_agent -- empty input short-circuit
# ---------------------------------------------------------------------------

def test_empty_merged_skips_llm_and_returns_merged_only(monkeypatch):
    def fail_if_called(router, model_name, **kwargs):
        raise AssertionError("LLM should not be called for an empty merged query")

    monkeypatch.setattr(fallback_module.client_llm, "get_model", fail_if_called)

    result = query_decomposer_agent({"merged": ""})

    assert result == {"sub_queries": [""]}


# ---------------------------------------------------------------------------
# query_decomposer_agent -- success path
# ---------------------------------------------------------------------------

def test_success_returns_normalized_sub_queries_with_merged_first(monkeypatch):
    merged = "What safety procedures are required for unloading a load of sand from a transport truck?"
    proposed = QueryDecomposition(
        sub_queries=[
            "employees stand clear of vehicle being loaded or unloaded",
            "motor vehicle obstructed view to the rear reverse signal alarm",
        ]
    )
    _patch_get_model(monkeypatch, _FakeChatModel(result=proposed))

    result = query_decomposer_agent({"merged": merged})

    assert result["sub_queries"][0] == merged
    assert "employees stand clear of vehicle being loaded or unloaded" in result["sub_queries"]
    assert "decomposer_error" not in result


# ---------------------------------------------------------------------------
# query_decomposer_agent -- failure degrades to [merged], never []
# ---------------------------------------------------------------------------

def test_llm_failure_degrades_to_merged_only(monkeypatch):
    merged = "What safety procedures are required for electric arc welding?"
    _patch_get_model(monkeypatch, _FakeChatModel(fail_message="groq down"))

    result = query_decomposer_agent({"merged": merged})

    assert result["sub_queries"] == [merged]
    assert result["sub_queries"] != []
    assert "groq down" in result["decomposer_error"]
