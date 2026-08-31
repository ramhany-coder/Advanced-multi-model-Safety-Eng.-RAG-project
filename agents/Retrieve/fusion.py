"""Reciprocal Rank Fusion for the reranker node (agents/Reranker/agent.py).

Similarity scores from different sub-queries aren't comparable -- 0.8 against
a narrow sub-query says nothing about 0.7 against the original question.
RRF sidesteps that by fusing on rank position instead of score, so a chunk
that surfaces for several sub-queries outranks one that merely tops a single
list, deterministically and with no LLM call.
"""
import json

from config import settings

RRF_K = 60


def _load_retrieval_weights(path: str) -> dict[str, float]:
    """
    Build the chunk_id -> retrieval_weight map once at import time from the
    same corpus file the retriever loads (chunk_corpus.py tags every chunk:
    operative 1.0, appendix_mandatory 0.9, scope/definitions 0.5,
    appendix_nonmandatory/administrative 0.3).

    agents/Retrieve/embedding_cache.py's load_parent_documents() doesn't
    carry retrieval_weight into Document.metadata (that metadata is already
    baked into the persisted Chroma cache), so this reads the corpus file
    directly instead of relying on it showing up on retrieved Documents.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    items = registry.values() if isinstance(registry, dict) else registry
    weights: dict[str, float] = {}
    for item in items:
        chunk_id = item.get("chunk_id")
        if chunk_id is not None and "retrieval_weight" in item:
            weights[chunk_id] = item["retrieval_weight"]
    return weights


_RETRIEVAL_WEIGHTS = _load_retrieval_weights(
    settings.PARENT_PATH or "parent_store/registry.json"
)


def _chunk_id(doc) -> object:
    md = getattr(doc, "metadata", None) or {}
    return md.get("chunk_id") or getattr(doc, "id", None) or id(doc)


def rrf_fuse(ranked_lists, top_k=8, k=RRF_K, weights=None):
    """
    Fuse per-sub-query result lists into one ordered list.

    ranked_lists : one list of Documents per sub-query, each already in that
                   query's own relevance order.
    weights      : optional multiplier keyed by list position as a string.
                   Give the original query more pull than a narrow sub-query -
                   it has repeatedly found paragraphs no decomposition asked
                   for, such as 1926.57's "unloading shipments of sand".

    Each candidate's fused RRF score is then multiplied by its
    retrieval_weight (looked up by chunk_id in _RETRIEVAL_WEIGHTS, default
    1.0 -- neutral -- for anything not found there) before the final top_k
    cut, so operative text outranks a definitions block that merely shares
    vocabulary with the query.
    """
    scores, best = {}, {}
    for i, docs in enumerate(ranked_lists):
        w = (weights or {}).get(str(i), 1.0)
        for rank, doc in enumerate(docs):
            cid = _chunk_id(doc)
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
            best.setdefault(cid, doc)

    for cid in scores:
        scores[cid] *= _RETRIEVAL_WEIGHTS.get(cid, 1.0)

    order = sorted(scores, key=lambda c: -scores[c])
    return [best[c] for c in order[:top_k]]
