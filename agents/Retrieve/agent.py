
from config import settings
from agents.helpers import clamp_text
from agents.Retrieve.helpers import load_parent_documents, _ensemble_retrieve, _route_rerank


def hyb_retriver_agent(state) -> dict:
    query = clamp_text(state.get("merged") or "")
    k = int(state.get("k") or 5)
    section_ids = state.get("section_ids") or []

    fetch_k = max(10, k * 3)

    # Step 1: load the full OSHA section documents (parents), optionally filtered by section_id.
    parent_docs = load_parent_documents(
        registry_path=settings.PARENT_PATH or "parent_store/registry.json",
        given_section_id=section_ids or None,
    )

    try:
        # Step 2: BM25 + semantic retrieval directly over the parent documents, fused via ensemble.
        hits = _ensemble_retrieve(parent_docs, query, fetch_k)
    except Exception as e:
        return {
            "context": [],
            "retrieval_mode": "parent_retrieval_failed",
            "bm25_error": str(e),
        }

    # Step 3: rerank the parent documents and return them.
    reranked_docs, rerank_mode = _route_rerank(hits, query, k)

    return {
        "context": reranked_docs,
        "retrieval_mode": f"ensemble_parent_retrieval+{rerank_mode}",
    }
