import json
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_classic.retrievers import EnsembleRetriever
except Exception:
    EnsembleRetriever = None
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from agents.Retrieve.reranker_pachage import pinecone_client
from agents.Retrieve.prompts import rerank_system_prompt, rerank_human_prompt
from agents.Retrieve.schemas import RerankSelection
from agents.fallback import FallBack
from agents.Retrieve.bm25 import BM25


def load_parent_docstore(
    registry_path: str = "parent_store/registry.json",
    given_section_id: list[str] = None
):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    parent_docstore = InMemoryStore()
    items = []

    if isinstance(registry, dict):

        for doc_id, item in registry.items():

            section_id = item.get("section_id") or str(doc_id)
            title = item.get("title") or ""
            full_text = item.get("full_text") or ""

            if given_section_id is not None and section_id not in given_section_id:
                continue

            page_content = (
                f"Title: {title}\n\n"
                f"Full Text:\n{full_text}"
            )

            metadata = {
                "doc_id": str(doc_id),
                "section_id": section_id,
                "title": title,
            }

            document = Document(
                page_content=page_content,
                metadata=metadata,
            )

            items.append(
                (
                    str(doc_id),
                    document,
                )
            )

    elif isinstance(registry, list):

        for i, item in enumerate(registry):

            section_id = (
                item.get("section_id")
                or item.get("doc_id")
                or item.get("parent_index")
                or str(i)
            )

            if given_section_id is not None and section_id not in given_section_id:
                continue

            title = item.get("title") or ""
            full_text = item.get("full_text") or ""

            page_content = (
                f"Title: {title}\n\n"
                f"Full Text:\n{full_text}"
            )

            metadata = {
                "doc_id": str(section_id),
                "section_id": section_id,
                "title": title,
            }

            document = Document(
                page_content=page_content,
                metadata=metadata,
            )

            items.append(
                (
                    str(section_id),
                    document,
                )
            )

    else:
        raise TypeError(
            "registry.json must be either dict or list."
        )

    parent_docstore.mset(items)

    return parent_docstore


# Ensemble fusion weights for combining semantic (dense) + BM25 (sparse) child retrieval.
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# Router: which LLM answers the rerank fallback when Pinecone isn't configured.
RERANK_PRIMARY_ROUTER = "groq"
RERANK_PRIMARY_MODEL = "llama-3.1-8b-instant"
RERANK_SECONDARY_ROUTER = "gpt"
RERANK_SECONDARY_MODEL = "gpt-4o-mini"
RERANK_FALLBACK_ORDER = [RERANK_PRIMARY_ROUTER, RERANK_SECONDARY_ROUTER]

rerank_llm = FallBack(
    **{
        f"llm_{RERANK_PRIMARY_ROUTER}": RERANK_PRIMARY_MODEL,
        f"llm_{RERANK_SECONDARY_ROUTER}": RERANK_SECONDARY_MODEL,
    }
)


def _build_embedder() -> HuggingFaceEmbeddings:
    model_path = settings.EMBEDDING_MODEL_PATH
    model_name = model_path if model_path and os.path.exists(model_path) else (
        settings.EMBEDDING_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


emb = _build_embedder()


def _combined_title_and_text(children: list[Document]) -> list[Document]:
    """Re-shape child docs so BM25 and semantic search both score title + full text."""
    return [
        Document(
            page_content=f"Title: {child.metadata.get('title', '')}\n\nFull Text:\n{child.page_content}",
            metadata=child.metadata,
        )
        for child in children
    ]


def _ensemble_child_retrieve(children: list[Document], query: str, fetch_k: int) -> list[Document]:
    """Steps 2+3: BM25 and semantic retrieval over the child docs' title + full text, fused 0.6/0.4."""
    if not children:
        return []

    combined = _combined_title_and_text(children)

    dense_store = Chroma.from_documents(combined, embedding=emb)
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": fetch_k})

    sparse_retriever = BM25(combined, k=fetch_k).retriever

    if EnsembleRetriever is None:
        return dense_retriever.invoke(query)[:fetch_k]

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return ensemble.invoke(query)[:fetch_k]


def _children_to_parents(
    child_docs: list[Document],
    parent_docstore,
    max_parents: int,
) -> list[Document]:
    """Step 4: map matched child sections back to their full parent OSHA section documents."""
    parent_docs = []
    seen_doc_ids = set()

    for child in child_docs:
        doc_id = child.metadata.get("doc_id")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        parent = parent_docstore.mget([doc_id])[0]
        if parent is None:
            continue

        parent = Document(page_content=parent.page_content, metadata=dict(parent.metadata or {}))
        parent.metadata["matched_child_section_id"] = child.metadata.get("section_id")
        parent_docs.append(parent)

        if len(parent_docs) >= max_parents:
            break

    return parent_docs


def _llm_rerank(documents: list[Document], query: str, k: int) -> tuple[list[Document], str]:
    if len(documents) <= 1:
        return documents[:k], "llm_rerank_skipped_single_doc"

    candidates_block = "\n".join(
        f"- section_id={doc.metadata.get('section_id', 'unknown')} "
        f"title={doc.metadata.get('title', '')}"
        for doc in documents
    )

    messages = [
        SystemMessage(content=rerank_system_prompt),
        HumanMessage(content=rerank_human_prompt(query, candidates_block, k)),
    ]

    try:
        result = rerank_llm.constrained_invoke(
            messages, RERANK_FALLBACK_ORDER, constraine_model=RerankSelection
        )
        ranked_ids = result["ranked_section_ids"]
        by_section_id = {str(doc.metadata.get("section_id")): doc for doc in documents}

        ranked_docs = [by_section_id[sid] for sid in ranked_ids if sid in by_section_id]
        for doc in documents:
            if doc not in ranked_docs:
                ranked_docs.append(doc)

        return ranked_docs[:k], "llm_rerank"
    except Exception:
        return documents[:k], "llm_rerank_failed_passthrough"


def _route_rerank(documents: list[Document], query: str, k: int) -> tuple[list[Document], str]:
    """Step 5 router: Pinecone rerank when PINECONE_API_KEY is configured, else an LLM call."""
    if not documents:
        return [], "no_candidates"

    if pinecone_client is not None:
        try:
            reranked = list(pinecone_client.rerank(documents, query))
            if reranked:
                return reranked[:k], "pinecone_rerank"
        except Exception:
            pass

    return _llm_rerank(documents, query, k)
