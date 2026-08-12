
from config import settings
from agents.helpers import clamp_text
from agents.Retrieve.helpers import load_parent_docstore , _ensemble_child_retrieve, _children_to_parents, _route_rerank
from agents.Retrieve.meta_data_filter import MetadataFilter


def hyb_retriver_agent(state) -> dict:
    query = clamp_text(state.get("merged") or "")
    k = int(state.get("k") or 5)
    section_ids = state.get("section_ids") or []
    
    fetch_k = max(10, k * 3)

    # Step 1: metadata filtering on section_id, over the child sections.
    metadata_filter = MetadataFilter(settings.CHILD_DOCUMENTS_PATH)
    child_candidates = metadata_filter.get_results(section_ids) if section_ids else []
    if not child_candidates:
        child_candidates = metadata_filter.get_all()

    try:
        # Steps 2 + 3: BM25 + semantic retrieval over the child docs, fused via ensemble.
        child_hits = _ensemble_child_retrieve(child_candidates, query, fetch_k)

        # Step 4: get the parent documents for the matched children.
        parent_docstore = load_parent_docstore(
            registry_path=settings.PARENT_PATH or "parent_store/registry.json",
        )
        parent_docs = _children_to_parents(child_hits, parent_docstore, max_parents=fetch_k)
    except Exception as e:
        return {
            "context": [],
            "retrieval_mode": "child_retrieval_failed",
            "bm25_error": str(e),
        }

    # Step 5: rerank the parent documents and return them.
    reranked_docs, rerank_mode = _route_rerank(parent_docs, query, k)

    return {
        "context": reranked_docs,
        "retrieval_mode": f"metadata_filter+ensemble_child_retrieval+{rerank_mode}",
    }
