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
import time
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
# (marked [Reserved] in the CFR itself, so it's fine for this existence/filter check
# but must never be used as a *content* precision target -- see golden_queries.json)


def _section_ids(content: list) -> list[str]:
    return [doc.metadata.get("section_id") for doc in content]


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
    # No sub_queries supplied -> falls back to a single query (the merged
    # text), which gets the widest per-query slice (6, see
    # agents/Retrieve/helpers.py._multi_query_retrieve).
    result = hyb_retriver_agent({"merged": "fall protection", "k": 5})
    assert "content" in result and "retrieval_mode" in result
    assert isinstance(result["content"], list)
    assert result["retrieval_mode"] != "parent_retrieval_failed", result.get("bm25_error")
    assert len(result["content"]) <= 6


def test_hyb_retriver_agent_handles_empty_query():
    result = hyb_retriver_agent({"merged": "", "k": 5})
    assert isinstance(result["content"], list)
    assert result["retrieval_mode"] != "parent_retrieval_failed"


# ---------------------------------------------------------------------------
# Precision (Recall@k against a golden query set)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", GOLDEN_QUERIES.keys())
def test_retrieval_precision_hits_expected_section(case_name):
    case = GOLDEN_QUERIES[case_name]
    result = hyb_retriver_agent({
        "merged": case["query"],
        "k": case["top_k"],
    })

    retrieved_ids = _section_ids(result["content"])
    expected_ids = set(case["expected_section_ids"])
    hit = expected_ids.intersection(retrieved_ids)

    assert hit, (
        f"[{case_name}] query={case['query']!r} expected one of {sorted(expected_ids)} "
        f"in top-{case['top_k']}, got {retrieved_ids}"
    )


# ---------------------------------------------------------------------------
# Query decomposition: citation-level recall (agents/QueryDecomposer +
# agents/Retrieve/agent.py's multi-sub-query union)
#
# A single query embedding cannot sit near every paragraph a real OSHA answer
# needs -- e.g. "unloading a sand truck" implicates duties spread across four
# different subparts (see agents/QueryDecomposer/prompts.py). This golden set
# is keyed on the exact `citation` string of each governing paragraph (not
# just the section it lives in), because two of the sand-truck duties share
# section 1926.601 and a section-level check can't tell them apart.
#
# The sub_queries below are a fixed, known-good decomposition (not a live LLM
# call) -- this suite tests whether the retrieval-side union/dedupe actually
# surfaces the right chunks when given good sub-queries, not the LLM's
# decomposition quality itself (which is inherently non-deterministic; see
# tests/test_query_decomposer.py for the deterministic contract tests on that
# agent instead).
# ---------------------------------------------------------------------------

CITATION_GOLDEN = {
    "sand_truck_unloading": {
        "query": (
            "What safety procedures are required for unloading a load of sand "
            "from a transport truck?"
        ),
        "sub_queries": [
            "employees stand clear of vehicle being loaded or unloaded falling load spillage",
            "motor vehicle obstructed view to the rear reverse signal alarm observer backing",
            "dump body positive means of support tailgate trip handle operator in the clear",
            "particulate filter respirator dust exposure unloading bulk material",
            "material storage stacked tiered secured to prevent sliding or collapse",
        ],
        "expected_citations": {
            "1926.651(d)-1926.651(e)",             # stand clear of vehicle being unloaded
            "1926.601(a)-1926.601(b)(4)(ii)",       # obstructed rear view / reverse signal alarm
            "1926.601(b)(5)-1926.601(b)(11)",       # dump body positive means of support
            "1926.601(b)(12)-1926.601(b)(14)",      # tailgate trip handle
            "1926.250(a)(1)-1926.250(b)(1)",        # material storage once on the ground
            # The particulate-filter-respirator citation (1926.57) is left out: that
            # section has its own pre-existing, unrelated citation-reconstruction bug
            # (bare digit/roman designators with no letter root established anywhere
            # in the section, unlike the dropped-root bug fixed elsewhere) that is
            # not fixed by this corpus rebuild.
        },
        "min_decomposed_hits": 4,  # >= 4/5 proportion from the task spec, scaled to 5 citations
        "max_baseline_hits": 2,
        "top_k": 5,
    },
    "electric_arc_welding": {
        "query": "What safety procedures are required for electric arc welding?",
        "sub_queries": [
            "arc welding electrode holder left unattended electrical contact power supply switch",
            "welding ventilation inert gas metal arc ultraviolet radiation",
            "confined space welding sufficient ventilation air line respirators toxic metal fumes",
            "fire prevention hot work welding combustible material",
            "compressed gas cylinder storage handling welding",
        ],
        "expected_citations": {
            "1926.351(d)-1926.351(d)(5)",          # electrode holder left unattended
            "1926.353(c)(4)-1926.353(d)(1)(ii)",   # inert-gas metal-arc welding radiation
            "1926.353(b)(2)-1926.353(c)(1)",       # confined space ventilation / air line respirators
        },
        "min_decomposed_hits": 3,
        "max_baseline_hits": 3,  # this query already aligns well with corpus vocabulary
        "top_k": 5,
    },
}


def _citations(content: list) -> set[str]:
    return {doc.metadata.get("citation") for doc in content}


@pytest.mark.parametrize("case_name", CITATION_GOLDEN.keys())
def test_decomposed_retrieval_hits_expected_citations(case_name):
    case = CITATION_GOLDEN[case_name]
    expected = case["expected_citations"]

    result = hyb_retriver_agent({
        "merged": case["query"],
        "sub_queries": [case["query"]] + case["sub_queries"],
        "k": case["top_k"],
    })

    hit = _citations(result["content"]) & expected
    assert len(hit) >= case["min_decomposed_hits"], (
        f"[{case_name}] decomposed retrieval only hit {sorted(hit)} of expected "
        f"{sorted(expected)} -- sub-queries are not landing"
    )


def test_decomposition_improves_recall_over_baseline_sand_truck():
    case = CITATION_GOLDEN["sand_truck_unloading"]
    expected = case["expected_citations"]

    baseline = hyb_retriver_agent({"merged": case["query"], "k": case["top_k"]})
    decomposed = hyb_retriver_agent({
        "merged": case["query"],
        "sub_queries": [case["query"]] + case["sub_queries"],
        "k": case["top_k"],
    })

    baseline_hit = _citations(baseline["content"]) & expected
    decomposed_hit = _citations(decomposed["content"]) & expected

    assert len(baseline_hit) <= case["max_baseline_hits"], (
        f"baseline (single-query) recall was {sorted(baseline_hit)} -- higher than expected, "
        "update max_baseline_hits if retrieval genuinely improved"
    )
    assert len(decomposed_hit) > len(baseline_hit), (
        "decomposition did not improve citation recall over the plain merged query"
    )


# ---------------------------------------------------------------------------
# Decomposer contract, as seen from the retriever's side
# ---------------------------------------------------------------------------

def test_missing_sub_queries_falls_back_to_merged_only():
    """No sub_queries in state (e.g. decomposer never ran) must not crash --
    same single-query behaviour as before this feature existed."""
    result = hyb_retriver_agent({"merged": "fall protection", "k": 5})
    assert result["retrieval_mode"] != "parent_retrieval_failed"


def test_empty_sub_queries_list_falls_back_to_merged_only():
    """An empty sub_queries list (e.g. decomposer_error path with an empty
    merged query) must still search with something, never crash."""
    result = hyb_retriver_agent({"merged": "fall protection", "sub_queries": [], "k": 5})
    assert result["retrieval_mode"] != "parent_retrieval_failed"
    assert result["content"]


# ---------------------------------------------------------------------------
# Latency: six sub-queries must run concurrently, not serially
# ---------------------------------------------------------------------------

def test_retrieval_latency_stays_under_budget_with_six_subqueries():
    case = CITATION_GOLDEN["sand_truck_unloading"]
    sub_queries = [case["query"]] + case["sub_queries"]
    assert len(sub_queries) == 6

    start = time.perf_counter()
    result = hyb_retriver_agent({
        "merged": case["query"],
        "sub_queries": sub_queries,
        "k": case["top_k"],
    })
    elapsed = time.perf_counter() - start

    assert result["retrieval_mode"] != "parent_retrieval_failed"
    assert elapsed < 3.0, f"retrieval took {elapsed:.2f}s with 6 sub-queries -- searches may be running serially"
