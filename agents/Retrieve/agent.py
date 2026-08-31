
from config import settings
from agents.helpers import clamp_text
from agents.Retrieve.helpers import load_parent_documents, _multi_query_retrieve


def hyb_retriver_agent(state) -> dict:
    merged = clamp_text(state.get("merged") or "")

    # Search with every sub-query the decomposer produced, union the hits.
    # sub_queries[0] is always the merged query, so a missing/empty
    # sub_queries just degrades to today's single-query behaviour.
    sub_queries = state.get("sub_queries") or [merged]
    queries = [clamp_text(q) for q in sub_queries if q] or [merged]

    k = int(state.get("k") or 5)

    # "content" may already hold evidence carried over from a previous pass
    # (e.g. reranked evidence kept across a retry loop). New hits must be
    # appended to that, never replace it outright.

    fetch_k = max(10, k * 3)

    # Retrieval always runs over the full OSHA corpus.
    parent_docs = load_parent_documents(
        registry_path=settings.PARENT_PATH or "parent_store/registry.json",
    )

    try:
        # BM25 + semantic retrieval over the parent documents, fused via
        # ensemble, run once per sub-query concurrently and deduped by
        # chunk_id -- one duty missed by the merged query alone can still be
        # found by a sub-query aimed straight at it.
        hits = _multi_query_retrieve(parent_docs, queries, fetch_k)
    except Exception as e:
        return {
            "retrieval_mode": "parent_retrieval_failed",
            "bm25_error": str(e),
        }

    existing_content = list(state.get("content") or [])
    if not hits:
        return {
            "retrieval_mode": "ensemble_parent_retrieval+no_candidates",
        }

    return {
        "content": existing_content + hits,
        "retrieval_mode": "ensemble_parent_retrieval",
    }
