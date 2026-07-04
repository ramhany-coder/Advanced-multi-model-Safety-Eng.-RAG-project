import re
import json
import traceback
import os
import re
import json
import sys
import traceback
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_classic.retrievers import EnsembleRetriever
except Exception:
     EnsembleRetriever = None

try:
    from models import State
except Exception:
    class State(TypedDict, total=False):
        query: str
        merged: str
        k: int
        context: Any
        retrieval_mode: str


try:
    from langchain_pinecone import PineconeRerank
except Exception:
    PineconeRerank = None

try:
    import langchain_community.retrievers as community_retrievers
    sys.modules["langchain.retrievers"] = community_retrievers
except Exception:
    pass

try:
    from bm25_retriever.retriever import PersistentBM25Retriever
except Exception:
    PersistentBM25Retriever = None


LOCAL_EMBEDDING_MODEL = r"C:\AI\models\all-MiniLM-L6-v2"

if os.path.exists(LOCAL_EMBEDDING_MODEL):
    emb = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
else:
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def clamp_text(text: str, max_chars: int = 4000) -> str:
    """
    Prevent very long queries from breaking retrievers/rerankers.
    """
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

def extract_osha_section(query: str) -> str | None:
    """
    Extract base OSHA section from queries like:
    1926.451(f)(3) -> 1926.451
    1926.502(d)(16) -> 1926.502

    Important:
    This should be used on the ORIGINAL user query only,
    not rewritten_query or merged, to avoid LLM hallucinated OSHA sections.
    """
    if not query:
        return None

    match = re.search(r"\b(1926\.\d+)(?:\([a-zA-Z0-9]+\))*", query)
    if match:
        return match.group(1)

    return None


def load_parent_registry(registry_path: str = "parent_store/registry.json") -> dict:
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parent_docstore(registry_path: str = "parent_store/registry.json"):
    """
    Load parent OSHA registry.

    Returns:
        parent_docstore:
            InMemoryStore for exact section fallback.

        items:
            list[(doc_id, Document)] for exact section lookup.

        registry:
            raw registry dict/list so we can fetch only matched subsection text.
    """
    registry = load_parent_registry(registry_path)

    parent_docstore = InMemoryStore()
    items = []

    if isinstance(registry, dict):
        iterable = registry.items()

        for doc_id, item in iterable:
            page_content = (
                item.get("page_content")
                or item.get("full_text")
                or item.get("text")
                or ""
            )

            metadata = item.get("metadata") or {}
            metadata = dict(metadata)

            metadata["doc_id"] = str(doc_id)

            if "section_id" not in metadata:
                metadata["section_id"] = item.get("section_id", "")

            if "title" not in metadata:
                metadata["title"] = item.get("title", "")

            if "source" not in metadata:
                metadata["source"] = item.get("source") or item.get("url", "")

            items.append(
                (
                    str(doc_id),
                    Document(
                        page_content=page_content,
                        metadata=metadata
                    )
                )
            )

    elif isinstance(registry, list):
        for i, item in enumerate(registry):
            doc_id = str(item.get("doc_id") or item.get("parent_index") or i)

            page_content = (
                item.get("page_content")
                or item.get("full_text")
                or item.get("text")
                or ""
            )

            metadata = dict(item)
            metadata["doc_id"] = doc_id

            if "section_id" not in metadata:
                metadata["section_id"] = item.get("section_id", "")

            if "title" not in metadata:
                metadata["title"] = item.get("title", "")

            if "source" not in metadata:
                metadata["source"] = item.get("source") or item.get("url", "")

            items.append(
                (
                    doc_id,
                    Document(
                        page_content=page_content,
                        metadata=metadata
                    )
                )
            )

    else:
        raise TypeError("registry.json must be either dict or list.")

    parent_docstore.mset(items)

    return parent_docstore, items, registry


def exact_section_lookup_from_items(
    query: str,
    parent_items: list[tuple[str, Document]]
) -> list[Document]:
    """
    If user mentions an exact OSHA section, return that parent section directly.

    Example:
        1926.451(f)(3) -> return parent doc with section_id 1926.451

    Note:
        Use this ONLY with original user query.
    """
    section = extract_osha_section(query)

    if not section:
        return []

    matches = []

    for _, doc in parent_items:
        meta = doc.metadata or {}
        section_id = str(meta.get("section_id", "")).strip()

        if section_id == section:
            matches.append(doc)

    return matches


def normalize_registry_to_dict(registry) -> dict:
    """
    Convert registry list/dict into dict[doc_id] = parent_data.
    This makes evidence fetching easier.
    """
    if isinstance(registry, dict):
        return registry

    if isinstance(registry, list):
        normalized = {}

        for i, item in enumerate(registry):
            doc_id = str(item.get("doc_id") or item.get("parent_index") or i)
            normalized[doc_id] = item

        return normalized

    return {}


def find_subsection_in_parent(parent: dict, subsection_id: str) -> dict | None:
    """
    Supports:
        subsections as dict:
            {"1926.351::p001": {...}}

        subsections as list:
            [{"subsection_id": "1926.351::p001", ...}]
    """
    if not parent or not subsection_id:
        return None

    subsections = parent.get("subsections", {})

    if isinstance(subsections, dict):
        return subsections.get(subsection_id)

    if isinstance(subsections, list):
        for subsection in subsections:
            if str(subsection.get("subsection_id", "")).strip() == subsection_id:
                return subsection

    return None


def fetch_evidence_from_registry(
    child_docs: list[Document],
    registry
) -> list[Document]:
    """
    Convert retrieved child chunks into small responder evidence.

    Important:
        This does NOT return the full OSHA parent document.
        It returns only the matched subsection original text.
    """
    registry_dict = normalize_registry_to_dict(registry)

    evidence_docs = []
    seen = set()

    for child in child_docs:
        child_meta = child.metadata or {}

        doc_id = str(child_meta.get("doc_id", "")).strip()
        subsection_id = str(child_meta.get("subsection_id", "")).strip()

        if not doc_id:
            continue

        dedupe_key = (
            doc_id,
            subsection_id or str(child_meta.get("chunk_id", ""))
        )

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        parent = registry_dict.get(doc_id)

        if not isinstance(parent, dict):
            continue

        parent_meta = parent.get("metadata") or {}

        section_id = (
            parent.get("section_id")
            or parent_meta.get("section_id")
            or child_meta.get("section_id", "")
        )

        title = (
            parent.get("title")
            or parent_meta.get("title")
            or child_meta.get("title", "")
        )

        source = (
            parent.get("source")
            or parent.get("url")
            or parent_meta.get("source")
            or parent_meta.get("url")
            or child_meta.get("source", "")
        )

        subsection = find_subsection_in_parent(parent, subsection_id)

        text = ""
        summary = ""
        heading = child_meta.get("heading", "")

        if subsection:
            text = (
                subsection.get("text")
                or subsection.get("full_text")
                or subsection.get("page_content")
                or ""
            )

            summary = (
                subsection.get("summary")
                or subsection.get("extractive_summary")
                or ""
            )

            heading = subsection.get("heading", heading)

        # fallback مهم لو registry قديم ومفيهوش subsections
        # لكن الأفضل إن chunking الجديد يحط subsections في parent_store.
        if not text:
            text = child.page_content

        if not text:
            continue

        page_content = f"""
Section: {section_id}
Title: {title}
Subsection: {subsection_id}
Heading: {heading}

Original OSHA Text:
{text}

Summary:
{summary}
""".strip()

        evidence_docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "subsection_id": subsection_id,
                    "title": title,
                    "heading": heading,
                    "source": source,
                    "matched_child_chunk_id": child_meta.get("chunk_id"),
                    "matched_child_chunk_type": child_meta.get("chunk_type"),
                    "evidence_type": "subsection_original_text"
                }
            )
        )

    return evidence_docs


def safe_rerank(documents, query: str, k: int):
    """
    Rerank documents if PineconeRerank is available.
    Otherwise return the first k documents.
    """
    if not documents:
        return []

    k = max(1, int(k or 1))

    if PineconeRerank is None:
        return documents[:k]

    try:
        reranker = PineconeRerank(
            model="bge-reranker-v2-m3",
            top_n=k
        )

        return reranker.compress_documents(
            documents=documents,
            query=query
        )

    except Exception as e:
        print("Pinecone rerank failed:", repr(e))
        return documents[:k]


def hyb_retriver_agent(state: State) -> dict:
    """
    Hybrid retriever:

    1. Exact OSHA section lookup uses ORIGINAL query only.
    2. Dense/BM25 retrieval uses merged query.
    3. Returns subsection evidence, not full parent document.
    """
    original_query = state.get("query") or ""
    retrieval_query = state.get("merged") or original_query

    exact_query = clamp_text(original_query)
    retrieval_query = clamp_text(retrieval_query)

    k = int(state.get("k", 5) or 5)

    # Keep responder context small
    top_n = min(4, max(2, k))

    # Load parent registry once
    parent_docstore, parent_items, parent_registry = load_parent_docstore(
        "parent_store/registry.json"
    )

    # 1) Exact section lookup FIRST
    # Important:
    # Use original query, not merged/rewrite.
    exact_docs = exact_section_lookup_from_items(exact_query, parent_items)

    if exact_docs:
        reranked_response = safe_rerank(
            exact_docs,
            exact_query,
            min(top_n, len(exact_docs))
        )

        return {
            "context": reranked_response,
            "retrieval_mode": "exact_section_lookup",
            "matched_section": extract_osha_section(exact_query),
            "top_n": top_n
        }

    # 2) Load Chroma child chunk store
    vbd_ret = Chroma(
        collection_name="production_parent_child_store",
        embedding_function=emb,
        persist_directory="osha"
    )

    # 3) Dense retriever over child chunks
    dense_ret = vbd_ret.as_retriever(
        search_kwargs={"k": max(20, k * 4)}
    )

    # 4) If BM25 is unavailable, fallback to dense retrieval
    if PersistentBM25Retriever is None:
        child_docs = dense_ret.invoke(retrieval_query)

        evidence_docs = fetch_evidence_from_registry(
            child_docs=child_docs,
            registry=parent_registry,
            max_items=top_n
        )

        reranked_response = safe_rerank(
            evidence_docs,
            retrieval_query,
            top_n
        )

        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_evidence",
            "top_n": top_n
        }

    try:
        # 5) Load sparse BM25 retriever
        sparse_ret = PersistentBM25Retriever.load(save_dir="osha_sparse")
        sparse_ret.k = max(10, k * 2)

        # 6) Hybrid ensemble
        hybrid_ret = EnsembleRetriever(
            retrievers=[dense_ret, sparse_ret],
            weights=[0.55, 0.45]
        )

        retrieved_docs = hybrid_ret.invoke(retrieval_query)

        evidence_docs = fetch_evidence_from_registry(
            child_docs=retrieved_docs,
            registry=parent_registry,
            max_items=top_n
        )

        reranked_response = safe_rerank(
            evidence_docs,
            retrieval_query,
            top_n
        )

        return {
            "context": reranked_response,
            "retrieval_mode": "hybrid_bm25_child_to_evidence",
            "top_n": top_n
        }

    except Exception as e:
        print("BM25 load/retrieval failed:", repr(e))
        print(traceback.format_exc())

        child_docs = dense_ret.invoke(retrieval_query)

        evidence_docs = fetch_evidence_from_registry(
            child_docs=child_docs,
            registry=parent_registry,
            max_items=top_n
        )

        reranked_response = safe_rerank(
            evidence_docs,
            retrieval_query,
            top_n
        )

        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_evidence_after_bm25_error",
            "bm25_error": repr(e),
            "top_n": top_n
        }