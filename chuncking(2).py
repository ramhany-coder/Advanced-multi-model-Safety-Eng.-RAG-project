import os
import json
import re
import shutil
import argparse
import traceback
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from bm25_retriever.retriever import PersistentBM25Retriever
except Exception:
    PersistentBM25Retriever = None


STOP_WORDS = {
    "the", "and", "a", "of", "to", "in", "is", "that", "for", "on", "with",
    "as", "by", "an", "at", "or", "be", "are", "from", "this", "which"
}


def clean_text(text):
    text = text or ""
    return re.sub(r"\s+", " ", text).strip()


def clean_and_deduplicate_text(text):
    words = re.findall(r"\b\w+\b", (text or "").lower())
    seen = set()
    output = []

    for word in words:
        if word in STOP_WORDS or word.isdigit() or word in seen:
            continue
        seen.add(word)
        output.append(word)

    return " ".join(output)


def split_text(text, chunk_size=1200, overlap=200):
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def load_osha_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for i, item in enumerate(data):
        full_text = clean_text(item.get("full_text", ""))

        if not full_text:
            continue

        docs.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": item.get("url", ""),
                    "section_id": item.get("section_id", ""),
                    "title": item.get("title", ""),
                    "parent_index": i,
                },
            )
        )

    return docs


def load_summary_map(summary_path="osha_extractive_summaries.json"):
    path = Path(summary_path)

    if not path.exists():
        print(f"Summary JSON not found: {summary_path}")
        print("The script will create summary chunks from title + first text instead.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        summaries = json.load(f)

    summary_map = {}

    for item in summaries:
        doc_id = str(item.get("doc_id", ""))
        summary = (
            item.get("extractive_summary")
            or item.get("summary")
            or item.get("content")
            or ""
        )

        if doc_id and summary:
            summary_map[doc_id] = clean_text(summary)

    print(f"Loaded precomputed summaries: {len(summary_map)}")
    return summary_map


def reset_outputs():
    for folder in ["parent_store", "osha", "osha_sparse"]:
        if os.path.exists(folder):
            print(f"Deleting old folder: {folder}")
            shutil.rmtree(folder, ignore_errors=True)

    os.makedirs("parent_store", exist_ok=True)
    os.makedirs("osha", exist_ok=True)
    os.makedirs("osha_sparse", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-json", default="osha_extractive_summaries.json")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    print("\n=== OSHA RAG CHUNKING PIPELINE STARTED ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Input file: {args.input}")
    print(f"Summary file: {args.summary_json}")

    if args.reset:
        reset_outputs()
    else:
        os.makedirs("parent_store", exist_ok=True)
        os.makedirs("osha", exist_ok=True)
        os.makedirs("osha_sparse", exist_ok=True)

    print("\n[1/5] Loading OSHA documents...")
    docs = load_osha_json(args.input)
    print(f"Loaded documents: {len(docs)}")

    print("\n[2/5] Loading precomputed summaries...")
    summary_map = load_summary_map(args.summary_json)

    print("\n[3/5] Creating parent store, child chunks, summaries, and keywords...")
    parent_registry = {}
    child_docs = []
    sparse_docs = []

    for idx, doc in enumerate(docs):
        doc_id = str(idx)

        parent_registry[doc_id] = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }

        base_meta = {
            **doc.metadata,
            "doc_id": doc_id,
        }

        chunks = split_text(doc.page_content)

        for chunk_idx, chunk in enumerate(chunks):
            child_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **base_meta,
                        "chunk_type": "semantic_chunk",
                        "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    },
                )
            )

        summary = summary_map.get(doc_id)

        if not summary:
            summary = clean_text(
                f"{doc.metadata.get('title', '')}. "
                f"{doc.page_content[:1200]}"
            )

        child_docs.append(
            Document(
                page_content=(
                    f"OSHA section: {doc.metadata.get('section_id', '')}\n"
                    f"Title: {doc.metadata.get('title', '')}\n"
                    f"Summary:\n{summary}"
                ),
                metadata={
                    **base_meta,
                    "chunk_type": "precomputed_summary",
                    "chunk_id": f"{doc_id}_summary",
                },
            )
        )

        keyword_text = (
            f"OSHA {doc.metadata.get('section_id', '')}. "
            f"{doc.metadata.get('title', '')}. "
            f"Construction safety regulation standard."
        )

        child_docs.append(
            Document(
                page_content=keyword_text,
                metadata={
                    **base_meta,
                    "chunk_type": "metadata_keywords",
                    "chunk_id": f"{doc_id}_keywords",
                },
            )
        )

        sparse_docs.append(
            Document(
                page_content=clean_and_deduplicate_text(
                    f"{doc.metadata.get('section_id', '')} "
                    f"{doc.metadata.get('title', '')} "
                    f"{doc.page_content}"
                ),
                metadata=base_meta,
            )
        )

    print(f"Parent docs: {len(parent_registry)}")
    print(f"Child docs to embed: {len(child_docs)}")
    print(f"Sparse docs: {len(sparse_docs)}")

    with open("parent_store/registry.json", "w", encoding="utf-8") as f:
        json.dump(parent_registry, f, ensure_ascii=False, indent=2)

    print("Saved: parent_store/registry.json")

    print("About to load embedding model...", flush=True)

    embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\AI\models\all-MiniLM-L6-v2",  # local path
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})

    print("Embedding model loaded.", flush=True)

    print("\n[5/5] Building Chroma vector store...")
    vectorstore = Chroma(
        collection_name="production_parent_child_store",
        embedding_function=embeddings,
        persist_directory="osha",
    )

    batch_size = 64

    for start in range(0, len(child_docs), batch_size):
        end = min(start + batch_size, len(child_docs))
        vectorstore.add_documents(child_docs[start:end])
        print(f"Embedded child docs {start + 1}-{end} / {len(child_docs)}")

    print("Saved Chroma DB folder: osha")

    if not args.skip_bm25:
        if PersistentBM25Retriever is None:
            print("BM25 package not available. Skipping BM25 index.")
        else:
            print("\nBuilding BM25 sparse index...")
            PersistentBM25Retriever.from_documents(
                documents=sparse_docs,
                save_dir="osha_sparse",
            )
            print("Saved BM25 folder: osha_sparse")

    print("\n=== SUCCESS: OSHA RAG INDEX CREATED ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== SCRIPT FAILED ===")
        traceback.print_exc()
        input("\nPress Enter to close...")
        raise