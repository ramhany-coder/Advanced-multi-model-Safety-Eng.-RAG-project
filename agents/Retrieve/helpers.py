from concurrent.futures import ThreadPoolExecutor

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


def _build_retriever(documents: list[Document], fetch_k: int):
    """Build the fused dense+BM25 retriever once for a batch of searches, so
    a multi-sub-query pass doesn't rebuild the BM25 index (over the whole
    corpus) once per sub-query."""
    section_ids = list({d.metadata["section_id"] for d in documents})
    dense_retriever = dense_store.as_retriever(
        search_kwargs={"k": fetch_k, "filter": {"section_id": {"$in": section_ids}}}
    )

    sparse_retriever = BM25(documents, k=fetch_k).retriever

    if EnsembleRetriever is None:
        return dense_retriever

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )


def _ensemble_retrieve(documents: list[Document], query: str, fetch_k: int) -> list[Document]:
    """BM25 over the given parent documents, fused 0.6/0.4 with semantic search
    against the persisted OSHA embedding cache (restricted to those same documents)."""
    if not documents:
        return []

    retriever = _build_retriever(documents, fetch_k)
    return retriever.invoke(query)[:fetch_k]


def _multi_query_retrieve(
    documents: list[Document],
    queries: list[str],
    fetch_k: int,
) -> list[Document]:
    """
    Search once per sub-query against a single shared retriever, run
    concurrently, then union the hits deduped by chunk_id.

    The first query (the unmodified merged query, by convention) gets a
    wider per-query slice than the rest -- it's the one plain semantic match
    proven to surface paragraphs no decomposition thought to search for, so
    it keeps first priority.
    """
    if not documents or not queries:
        return []

    retriever = _build_retriever(documents, fetch_k)

    def _search(i: int, q: str) -> list[Document]:
        per_query_k = 6 if i == 0 else 4
        return retriever.invoke(q)[:per_query_k]

    with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as pool:
        results = list(pool.map(lambda pair: _search(*pair), enumerate(queries)))

    hits: list[Document] = []
    seen: set = set()
    for docs in results:
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            hits.append(doc)
    return hits
