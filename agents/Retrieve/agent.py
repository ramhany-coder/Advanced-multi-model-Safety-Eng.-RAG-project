from helpers import clamp_text
from langchain_chroma import Chroma
from reranker_pachage import pinecone_client
        
def load_parent_docstore(registry_path: str = "parent_store/registry.json"):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    parent_docstore = InMemoryStore()
    items = []

    if isinstance(registry, dict):
        for doc_id, item in registry.items():
            page_content = item.get("page_content") or item.get("full_text") or ""
            metadata = item.get("metadata") or {}
            metadata["doc_id"] = str(doc_id)
            items.append((
                str(doc_id),
                Document(page_content=page_content, metadata=metadata)
            ))

    elif isinstance(registry, list):
        for i, item in enumerate(registry):
            doc_id = str(item.get("doc_id") or item.get("parent_index") or i)
            page_content = item.get("page_content") or item.get("full_text") or ""
            metadata = dict(item)
            metadata["doc_id"] = doc_id
            items.append((
                doc_id,
                Document(page_content=page_content, metadata=metadata)
            ))

    else:
        raise TypeError("registry.json must be either dict or list.")

    parent_docstore.mset(items)
    return parent_docstore

def hyb_retriver_agent(state) -> dict:
    query = state.get("merged") or ""
    k = int(state.get("k", 5) or 5)

    # Protect query length (assuming clamp_text is imported/defined elsewhere)
    query = clamp_text(query)

    vbd_ret = Chroma(
        collection_name="production_parent_child_store",
        embedding_function=emb,
        persist_directory="osha"
    )

    parent_docstore = load_parent_docstore("parent_store/registry.json")

    dense_ret = vbd_ret.as_retriever(
        search_kwargs={"k": max(10, k*3)}
    )

    def children_to_parents(child_docs, max_parents):
        parent_docs = []
        seen_doc_ids = set()
        for child in child_docs:
            doc_id = child.metadata.get("doc_id")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            parent = parent_docstore.mget([doc_id])[0]
            if parent:
                parent.metadata = dict(parent.metadata or {})
                parent.metadata["matched_child_chunk_id"] = child.metadata.get("chunk_id")
                parent.metadata["matched_child_chunk_type"] = child.metadata.get("chunk_type")
                parent_docs.append(parent)
            if len(parent_docs) >= max_parents:
                break
        return parent_docs

    if PersistentBM25Retriever is None:
        child_docs = dense_ret.invoke(query)
        parent_docs = children_to_parents(child_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_parent"
        }

    try:
        sparse_ret = PersistentBM25Retriever.load(save_dir="osha_sparse")
        sparse_ret.k = max(5, k)

        hybrid_ret = EnsembleRetriever(
            retrievers=[dense_ret, sparse_ret],
            weights=[0.6, 0.4]
        )
        retrieved_docs = hybrid_ret.invoke(query)
        parent_docs = children_to_parents(retrieved_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "hybrid_child_to_parent"
        }

    except Exception as e:
        child_docs = dense_ret.invoke(query)
        parent_docs = children_to_parents(child_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_parent_after_bm25_error",
            "bm25_error": str(e)
        }