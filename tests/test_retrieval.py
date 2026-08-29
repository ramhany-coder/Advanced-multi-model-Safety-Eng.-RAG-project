"""Tests for the OSHA hybrid retrieval pipeline (agents/Retrieve/agent.py, helpers.py).

Covers two things, meant to run as a CI stage that gates merges to the retrieval code:

1. Functionality: parent-document loading, section_id filtering, and the
   hyb_retriver_agent node's contract (return shape, empty-input/empty-candidate
   edge cases) don't crash and behave as designed.
2. Precision: a small golden set of real OSHA queries (tests/fixtures/retrieval/
   golden_queries.json) with the section(s) known to be relevant. Each query must
   surface at least one relevant section within its top-k hybrid retrieval results
   (Recall@k / Hit@k). This is a regression guard against embedding model changes,
   fusion-weight changes, or reintroducing bugs like the old child/parent 1:1
   summary indirection -- not a claim that current retrieval is optimal.

Runs the real local embedding model (HuggingFace MiniLM, CPU) and real BM25 over
the actual parent_store/registry.json corpus -- no mocking. Expects to run with
the repo root as the working directory (the app's own default PARENT_PATH
assumes this).
"""
import json
from pathlib import Path

import pytest

from agents.Retrieve.agent import hyb_retriver_agent
from agents.Retrieve.helpers import load_parent_documents, _ensemble_retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "parent_store" / "registry.json"
STATS_PATH = REPO_ROOT / "parent_store" / "stats.json"

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "retrieval"
GOLDEN_QUERIES = json.loads((FIXTURES_DIR / "golden_queries.json").read_text(encoding="utf-8"))

KNOWN_SECTION_ID = "1926.22"  # Recording and reporting of injuries -- known to exist in registry.json


def _section_ids(context: list) -> list[str]:
    return [doc.metadata.get("section_id") for doc in context]


# ---------------------------------------------------------------------------
# Functionality
# ---------------------------------------------------------------------------

def test_load_parent_documents_returns_all_parents():
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    docs = load_parent_documents(registry_path=str(REGISTRY_PATH))
    assert len(docs) == stats["parent_docs"]
    assert all(doc.metadata.get("section_id") and doc.metadata.get("doc_id") for doc in docs)


def test_load_parent_documents_filters_by_section_id():
    docs = load_parent_documents(
        registry_path=str(REGISTRY_PATH),
        given_section_id=[KNOWN_SECTION_ID],
    )
    assert len(docs) == 1
    assert docs[0].metadata["section_id"] == KNOWN_SECTION_ID
    assert docs[0].page_content.strip()


def test_load_parent_documents_unknown_section_id_returns_empty():
    docs = load_parent_documents(
        registry_path=str(REGISTRY_PATH),
        given_section_id=["not-a-real-section-id"],
    )
    assert docs == []


def test_load_parent_documents_missing_registry_raises():
    with pytest.raises(FileNotFoundError):
        load_parent_documents(registry_path=str(REPO_ROOT / "no_such_registry.json"))


def test_ensemble_retrieve_empty_documents_returns_empty():
    assert _ensemble_retrieve([], "any query", fetch_k=5) == []


def test_hyb_retriver_agent_returns_expected_shape():
    result = hyb_retriver_agent({"merged": "fall protection", "k": 5, "section_ids": []})
    assert "context" in result and "retrieval_mode" in result
    assert isinstance(result["context"], list)
    assert result["retrieval_mode"] != "parent_retrieval_failed", result.get("bm25_error")
    assert len(result["context"]) <= 5


def test_hyb_retriver_agent_handles_empty_query():
    result = hyb_retriver_agent({"merged": "", "k": 5, "section_ids": []})
    assert isinstance(result["context"], list)
    assert result["retrieval_mode"] != "parent_retrieval_failed"


def test_hyb_retriver_agent_respects_section_id_filter():
    result = hyb_retriver_agent({
        "merged": "requirements",
        "k": 5,
        "section_ids": [KNOWN_SECTION_ID],
    })
    ids = _section_ids(result["context"])
    assert ids, "expected at least one hit when filtering to a known section_id"
    assert all(sid == KNOWN_SECTION_ID for sid in ids)


def test_hyb_retriver_agent_unknown_section_id_returns_no_candidates():
    result = hyb_retriver_agent({
        "merged": "fall protection",
        "k": 5,
        "section_ids": ["not-a-real-section-id"],
    })
    assert result["context"] == []
    assert result["retrieval_mode"].endswith("no_candidates")


# ---------------------------------------------------------------------------
# Precision (Recall@k against a golden query set)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", GOLDEN_QUERIES.keys())
def test_retrieval_precision_hits_expected_section(case_name):
    case = GOLDEN_QUERIES[case_name]
    result = hyb_retriver_agent({
        "merged": case["query"],
        "k": case["top_k"],
        "section_ids": [],
    })

    retrieved_ids = _section_ids(result["context"])
    expected_ids = set(case["expected_section_ids"])
    hit = expected_ids.intersection(retrieved_ids)

    assert hit, (
        f"[{case_name}] query={case['query']!r} expected one of {sorted(expected_ids)} "
        f"in top-{case['top_k']}, got {retrieved_ids}"
    )
