"""Embedding model + OSHA parent-doc Chroma cache.

Split out of helpers.py so the offline build script (scripts/build_osha_embeddings.py)
can call build_embedding_cache() without importing helpers.py, which eagerly loads
the cache at import time and is meant to fail fast if it's missing.
"""
import hashlib
import json
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from model_manager import ensure_model_downloaded, SENTENCE_TRANSFORMER_IGNORE_PATTERNS

DOC_COLLECTION_NAME = "osha_parent_docs"


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


def _build_embedder() -> HuggingFaceEmbeddings:
    # Shares the same local directory as Embedding_Model: downloads once, then
    # every later call (including across app restarts) loads it from disk.
    model_path = ensure_model_downloaded(
        settings.EMBEDDING_MODEL_NAME,
        settings.EMBEDDING_MODEL_PATH,
        ignore_patterns=SENTENCE_TRANSFORMER_IGNORE_PATTERNS,
    )
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )


emb = _build_embedder()


def _registry_fingerprint(registry_path: str) -> str:
    with open(registry_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _cache_meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "cache_meta.json")


def _load_cache_meta(cache_dir: str) -> dict | None:
    meta_path = _cache_meta_path(cache_dir)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache_meta(cache_dir: str, meta: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_meta_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def build_embedding_cache(registry_path: str | None = None) -> None:
    """Offline step: embed every OSHA parent document once and persist the
    Chroma store + cache_meta.json to settings.EMBEDDINGS_CACHE_DIR.

    Run via scripts/build_osha_embeddings.py, then commit the cache directory.
    The running app never calls this -- see load_dense_store() below."""
    cache_dir = settings.EMBEDDINGS_CACHE_DIR
    registry_path = registry_path or settings.PARENT_PATH or "parent_store/registry.json"
    fingerprint = _registry_fingerprint(registry_path)
    # Identity for the cache fingerprint, not the on-disk path: emb.model_name
    # is now a local directory (see _build_embedder), which would make every
    # cache built before that local-dir change look spuriously stale.
    model_name = settings.EMBEDDING_MODEL_NAME

    store = Chroma(
        collection_name=DOC_COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=cache_dir,
    )
    store.delete_collection()
    store = Chroma(
        collection_name=DOC_COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=cache_dir,
    )
    all_docs = load_parent_documents(registry_path)
    if all_docs:
        store.add_documents(all_docs)
    _save_cache_meta(cache_dir, {"registry_fingerprint": fingerprint, "model_name": model_name})


def load_dense_store() -> Chroma:
    """Load the OSHA parent-doc embedding cache that shipped with the repo.

    Never builds or repairs it -- if the cache is missing or stale this raises
    immediately so the app fails fast instead of silently re-embedding the
    corpus in production. Run scripts/build_osha_embeddings.py and commit the
    refreshed cache when this happens."""
    cache_dir = settings.EMBEDDINGS_CACHE_DIR
    meta = _load_cache_meta(cache_dir)
    if meta is None:
        raise RuntimeError(
            f"OSHA embedding cache not found at '{cache_dir}'. Run "
            "`python scripts/build_osha_embeddings.py` and commit the resulting "
            "cache directory -- the app does not build it itself."
        )

    registry_path = settings.PARENT_PATH or "parent_store/registry.json"
    fingerprint = _registry_fingerprint(registry_path)
    # Identity for the cache fingerprint, not the on-disk path: emb.model_name
    # is now a local directory (see _build_embedder), which would make every
    # cache built before that local-dir change look spuriously stale.
    model_name = settings.EMBEDDING_MODEL_NAME
    if meta.get("registry_fingerprint") != fingerprint or meta.get("model_name") != model_name:
        raise RuntimeError(
            f"OSHA embedding cache at '{cache_dir}' is stale (registry.json or the "
            "embedding model changed since it was built). Re-run "
            "`python scripts/build_osha_embeddings.py` and commit the refreshed cache."
        )

    return Chroma(
        collection_name=DOC_COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=cache_dir,
    )
