
from config import settings
from agents.helpers import clamp_text
from agents.Retrieve.helpers import load_parent_documents, _ensemble_retrieve


def hyb_retriver_agent(state) -> dict:
    query = clamp_text(state.get("merged") or "")
    k = int(state.get("k") or 5)
    section_ids = state.get("section_ids") or []

    # "context" may already hold evidence carried over from a previous pass
    # (e.g. the doc-ID mapper's "content" folded in by the reranker before a
    # retry). New hits must be appended to that, never replace it outright.
    existing_context = list(state.get("context") or [])

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
            "context": existing_context,
            "retrieval_mode": "parent_retrieval_failed",
            "bm25_error": str(e),
        }

    if not hits:
        return {
            "context": existing_context,
            "retrieval_mode": "ensemble_parent_retrieval+no_candidates",
        }

    return {
        "context": existing_context + hits[:k],
        "retrieval_mode": "ensemble_parent_retrieval",
    }
