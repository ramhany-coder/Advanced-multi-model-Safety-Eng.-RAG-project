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
from agents.Retrieve.reranker_pachage import pinecone_client
from agents.Retrieve.prompts import rerank_system_prompt, rerank_human_prompt
from agents.Retrieve.schemas import RerankSelection
from agents.fallback import FallBack
from agents.Retrieve.bm25 import BM25
from config import settings


def load_parent_documents(
    registry_path: str = "parent_store/registry.json",
    given_section_id: list[str] = None,
) -> list[Document]:
    """Load full OSHA section documents (parents) for direct retrieval."""
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if isinstance(registry, dict):
        entries = registry.items()
    elif isinstance(registry, list):
        entries = enumerate(registry)
    else:
        raise TypeError("registry.json must be either dict or list.")

    documents = []

    for key, item in entries:
        section_id = item.get("section_id") or str(key)

        if given_section_id is not None and section_id not in given_section_id:
            continue

        title = item.get("title") or ""
        full_text = item.get("full_text") or ""
        doc_id = str(item.get("doc_id") or key)

        document = Document(
            page_content=f"Title: {title}\n\nFull Text:\n{full_text}",
            metadata={
                "doc_id": doc_id,
                "section_id": section_id,
                "title": title,
            },
        )
        documents.append(document)

    return documents


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


def _ensemble_retrieve(documents: list[Document], query: str, fetch_k: int) -> list[Document]:
    """BM25 and semantic retrieval directly over the parent documents, fused 0.6/0.4."""
    if not documents:
        return []

    dense_store = Chroma.from_documents(documents, embedding=emb)
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": fetch_k})

    sparse_retriever = BM25(documents, k=fetch_k).retriever

    if EnsembleRetriever is None:
        return dense_retriever.invoke(query)[:fetch_k]

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return ensemble.invoke(query)[:fetch_k]


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
