try:
    from langchain_classic.retrievers import EnsembleRetriever
except Exception:
    EnsembleRetriever = None
from langchain_core.documents import Document
from agents.Retrieve.bm25 import BM25
from agents.Retrieve.embedding_cache import emb, load_parent_documents, load_dense_store

# Ensemble fusion weights for combining semantic (dense) + BM25 (sparse) child retrieval.
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# Loaded once at import time from the pre-built cache shipped with the repo/container
# (see scripts/build_osha_embeddings.py) -- the running app never embeds the corpus itself.
dense_store = load_dense_store()


def _ensemble_retrieve(documents: list[Document], query: str, fetch_k: int) -> list[Document]:
    """BM25 over the given parent documents, fused 0.6/0.4 with semantic search
    against the persisted OSHA embedding cache (restricted to those same documents)."""
    if not documents:
        return []

    section_ids = list({d.metadata["section_id"] for d in documents})
    dense_retriever = dense_store.as_retriever(
        search_kwargs={"k": fetch_k, "filter": {"section_id": {"$in": section_ids}}}
    )

    sparse_retriever = BM25(documents, k=fetch_k).retriever

    if EnsembleRetriever is None:
        return dense_retriever.invoke(query)[:fetch_k]

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return ensemble.invoke(query)[:fetch_k]
