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


# Keep the same output folder names used by the current project.
PARENT_STORE_DIR = "parent_store"
VECTOR_STORE_DIR = "osha"
SPARSE_STORE_DIR = "osha_sparse"
CHROMA_COLLECTION_NAME = "production_parent_child_store"

STOP_WORDS = {
    "the", "and", "a", "of", "to", "in", "is", "that", "for", "on", "with",
    "as", "by", "an", "at", "or", "be", "are", "from", "this", "which",
    "shall", "must", "may", "osha", "section", "standard", "number", "title"
}


def clean_text(text: str) -> str:
    text = text or ""
    return re.sub(r"\s+", " ", text).strip()


def split_raw_paragraphs(text: str) -> list[str]:
    """
    Keep paragraph boundaries from the OSHA raw text.
    Your current OSHA JSON mostly does not preserve official paragraph labels such
    as 1926.351(a), so these paragraphs become synthetic child subsections.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [clean_text(p) for p in re.split(r"\n\s*\n+", text)]
    return [p for p in parts if p]


def extract_field(full_text: str, label: str) -> str:
    """
    Extract simple header fields like:
    Part Number:1926
    Subpart:1926 Subpart J
    Title:Arc welding and cutting.
    """
    pattern = rf"{re.escape(label)}\s*:\s*(.+)"
    match = re.search(pattern, full_text or "", flags=re.I)
    if not match:
        return ""
    return clean_text(match.group(1).split("\n")[0])


def extract_header_metadata(item: dict, doc_index: int) -> dict:
    full_text = item.get("full_text", "") or ""

    section_id = clean_text(item.get("section_id", "")) or extract_field(full_text, "Standard Number")
    title = clean_text(item.get("title", "")) or extract_field(full_text, "Title")

    part = extract_field(full_text, "Part Number")
    part_title = extract_field(full_text, "Part Number Title")
    subpart = extract_field(full_text, "Subpart")
    subpart_title = extract_field(full_text, "Subpart Title")
    gpo_source = extract_field(full_text, "GPO Source")

    # Fallback: infer part from section id, e.g. 1926.351 -> 1926
    if not part and section_id:
        part = section_id.split(".")[0].strip()

    return {
        "doc_id": str(doc_index),
        "parent_index": doc_index,
        "part": part,
        "part_title": part_title,
        "subpart": subpart,
        "subpart_title": subpart_title,
        "section_id": section_id,
        "title": title,
        "source": item.get("url", "") or item.get("source", ""),
        "gpo_source": gpo_source,
    }


def get_body_paragraphs(full_text: str) -> list[str]:
    """
    Return paragraphs after the OSHA header.
    Header ends at "GPO Source:e-CFR" in the current raw JSON.
    """
    paragraphs = split_raw_paragraphs(full_text)

    body_start = 0
    for i, p in enumerate(paragraphs):
        if p.lower().startswith("gpo source"):
            body_start = i + 1
            break

    body = paragraphs[body_start:]

    # Remove empty or repeated obvious header lines if any remain.
    skip_prefixes = (
        "part number:",
        "part number title:",
        "subpart:",
        "subpart title:",
        "standard number:",
        "title:",
        "gpo source:",
    )
    body = [p for p in body if not p.lower().startswith(skip_prefixes)]

    return body


def looks_like_heading(text: str) -> bool:
    """
    Heuristic only. The current raw OSHA JSON often strips official paragraph ids,
    leaving short heading-like lines such as "Welding cables and connectors."
    """
    text = clean_text(text)
    if not text:
        return False

    words = text.split()
    if len(words) > 10:
        return False

    if any(token in text.lower() for token in ["shall", "must", "employer", "employee", "required"]):
        return False

    # Short title-cased phrase or a short phrase ending with a period.
    return True


def short_summary(text: str, max_words: int = 45) -> str:
    text = clean_text(text)
    if not text:
        return ""

    # Prefer the first sentence if it is not too long.
    sentence_match = re.match(r"(.+?[.!?])(\s|$)", text)
    if sentence_match:
        first_sentence = clean_text(sentence_match.group(1))
        if 5 <= len(first_sentence.split()) <= max_words:
            return first_sentence

    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def normalize_keyword_text(text: str) -> str:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b", (text or "").lower())
    seen = set()
    output = []

    for word in words:
        if word in STOP_WORDS or word in seen:
            continue
        seen.add(word)
        output.append(word)

    return " ".join(output)


def build_subsections(section_id: str, full_text: str) -> dict:
    """
    Build child evidence units. Because the uploaded raw OSHA JSON does not
    reliably preserve official paragraph labels like (a), (b), (c), we keep
    stable synthetic ids:
        1926.351::p001
        1926.351::p002

    If you later scrape official paragraph ids, this function is the only part
    you need to replace.
    """
    body_paragraphs = get_body_paragraphs(full_text)

    if not body_paragraphs:
        return {
            f"{section_id}::p001": {
                "subsection_id": f"{section_id}::p001",
                "official_subsection_id": "",
                "level": 1,
                "ordinal": 1,
                "heading": "Full section",
                "text": clean_text(full_text),
                "summary": short_summary(full_text),
                "source_type": "synthetic_paragraph",
            }
        }

    subsections = {}
    pending_heading = ""

    ordinal = 0
    for paragraph in body_paragraphs:
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue

        # If the paragraph is only a heading, keep it as its own searchable
        # evidence unit. This avoids dropping OSHA headings that have no body
        # text in the current scraped format.
        if looks_like_heading(paragraph) and len(paragraph.split()) <= 7:
            pending_heading = paragraph.rstrip(".")
            ordinal += 1
            subsection_id = f"{section_id}::p{ordinal:03d}"
            subsections[subsection_id] = {
                "subsection_id": subsection_id,
                "official_subsection_id": "",
                "level": 1,
                "ordinal": ordinal,
                "heading": pending_heading,
                "text": paragraph,
                "summary": pending_heading,
                "source_type": "synthetic_heading",
            }
            continue

        ordinal += 1
        subsection_id = f"{section_id}::p{ordinal:03d}"

        heading = ""
        # Extract heading from "Heading. Body..." form.
        first_sentence = paragraph.split(".", 1)[0].strip()
        if 1 <= len(first_sentence.split()) <= 8 and len(paragraph.split(".")) > 1:
            maybe_heading = first_sentence
            if looks_like_heading(maybe_heading):
                heading = maybe_heading

        if not heading and pending_heading:
            heading = pending_heading

        subsections[subsection_id] = {
            "subsection_id": subsection_id,
            "official_subsection_id": "",
            "level": 1,
            "ordinal": ordinal,
            "heading": heading,
            "text": paragraph,
            "summary": short_summary(paragraph),
            "source_type": "synthetic_paragraph",
        }

    return subsections


def load_osha_json(path: str) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        full_text = item.get("full_text", "") or ""
        if not clean_text(full_text):
            continue

        meta = extract_header_metadata(item, i)
        docs.append({
            **meta,
            "full_text": full_text,
        })

    return docs


def load_summary_map(summary_path: str = "osha_extractive_summaries.json") -> dict:
    path = Path(summary_path)

    if not path.exists():
        print(f"Summary JSON not found: {summary_path}")
        print("The script will create section summaries from title + first text instead.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        summaries = json.load(f)

    summary_map = {}

    for item in summaries:
        doc_id = str(item.get("doc_id", ""))
        section_id = clean_text(item.get("section_id", ""))
        summary = (
            item.get("extractive_summary")
            or item.get("summary")
            or item.get("content")
            or ""
        )

        summary = clean_text(summary)

        if doc_id and summary:
            summary_map[doc_id] = summary

        # Extra fallback by section_id in case doc ids change later.
        if section_id and summary:
            summary_map[section_id] = summary

    print(f"Loaded precomputed summaries: {len(summary_map)}")
    return summary_map


def reset_outputs():
    for folder in [PARENT_STORE_DIR, VECTOR_STORE_DIR, SPARSE_STORE_DIR]:
        if os.path.exists(folder):
            print(f"Deleting old folder: {folder}")
            shutil.rmtree(folder, ignore_errors=True)

    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(SPARSE_STORE_DIR, exist_ok=True)


def build_dense_subsection_page(parent: dict, subsection: dict) -> str:
    keyword_seed = normalize_keyword_text(
        " ".join([
            parent.get("section_id", ""),
            parent.get("title", ""),
            parent.get("subpart_title", ""),
            subsection.get("heading", ""),
            subsection.get("summary", ""),
            subsection.get("text", ""),
        ])
    )

    return (
        f"OSHA section: {parent.get('section_id', '')}\n"
        f"Title: {parent.get('title', '')}\n"
        f"Subpart: {parent.get('subpart', '')} - {parent.get('subpart_title', '')}\n"
        f"Subsection: {subsection.get('subsection_id', '')}\n"
        f"Heading: {subsection.get('heading', '')}\n"
        f"Summary: {subsection.get('summary', '')}\n"
        f"Keywords: {keyword_seed}"
    ).strip()


def build_sparse_subsection_page(parent: dict, subsection: dict) -> str:
    """
    BM25 should see the original OSHA paragraph text, not summary only.
    This keeps lexical recall strong.
    """
    return clean_text(
        f"OSHA {parent.get('section_id', '')} "
        f"{parent.get('title', '')} "
        f"{parent.get('subpart_title', '')} "
        f"{subsection.get('heading', '')} "
        f"{subsection.get('summary', '')} "
        f"{subsection.get('text', '')}"
    )


def safe_meta(meta: dict) -> dict:
    """
    Chroma metadata values must be simple scalar values.
    """
    output = {}
    for key, value in meta.items():
        if value is None:
            output[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            output[key] = value
        else:
            output[key] = str(value)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-json", default="osha_extractive_summaries.json")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument(
        "--embedding-model",
        default=r"C:\AI\models\all-MiniLM-L6-v2",
        help="Local or HuggingFace embedding model path/name."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("\n=== OSHA RAG PARENT/SUBSECTION PIPELINE STARTED ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Input file: {args.input}")
    print(f"Summary file: {args.summary_json}")
    print(f"Output folders kept as: {PARENT_STORE_DIR}, {VECTOR_STORE_DIR}, {SPARSE_STORE_DIR}")

    if args.reset:
        reset_outputs()
    else:
        os.makedirs(PARENT_STORE_DIR, exist_ok=True)
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(SPARSE_STORE_DIR, exist_ok=True)

    print("\n[1/6] Loading OSHA raw documents...")
    docs = load_osha_json(args.input)
    print(f"Loaded documents: {len(docs)}")

    print("\n[2/6] Loading precomputed section summaries...")
    summary_map = load_summary_map(args.summary_json)

    print("\n[3/6] Building parent registry and child subsection chunks...")
    parent_registry = {}
    section_index = {}
    child_docs = []
    sparse_docs = []
    total_subsections = 0

    for doc in docs:
        doc_id = str(doc["doc_id"])
        section_id = doc.get("section_id", "")

        section_summary = (
            summary_map.get(doc_id)
            or summary_map.get(section_id)
            or short_summary(f"{doc.get('title', '')}. {doc.get('full_text', '')}", max_words=80)
        )

        subsections = build_subsections(section_id=section_id, full_text=doc.get("full_text", ""))

        parent = {
            "doc_id": doc_id,
            "part": doc.get("part", ""),
            "part_title": doc.get("part_title", ""),
            "subpart": doc.get("subpart", ""),
            "subpart_title": doc.get("subpart_title", ""),
            "section_id": section_id,
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "gpo_source": doc.get("gpo_source", ""),
            "full_text": doc.get("full_text", ""),
            "section_summary": section_summary,
            "subsections": subsections,
            "metadata": {
                "doc_id": doc_id,
                "parent_index": doc.get("parent_index", ""),
                "part": doc.get("part", ""),
                "subpart": doc.get("subpart", ""),
                "subpart_title": doc.get("subpart_title", ""),
                "section_id": section_id,
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
            },
        }

        parent_registry[doc_id] = parent
        if section_id:
            section_index[section_id] = doc_id

        base_meta = {
            "doc_id": doc_id,
            "part": parent.get("part", ""),
            "subpart": parent.get("subpart", ""),
            "subpart_title": parent.get("subpart_title", ""),
            "section_id": section_id,
            "title": parent.get("title", ""),
            "source": parent.get("source", ""),
        }

        # Section-level routing chunk. If selected later, agents.py expands it
        # into only a few top subsection evidence units, not the full document.
        child_docs.append(
            Document(
                page_content=(
                    f"OSHA section: {section_id}\n"
                    f"Title: {parent.get('title', '')}\n"
                    f"Subpart: {parent.get('subpart', '')} - {parent.get('subpart_title', '')}\n"
                    f"Section summary: {section_summary}"
                ),
                metadata=safe_meta({
                    **base_meta,
                    "subsection_id": "__section_summary__",
                    "chunk_type": "section_summary",
                    "chunk_id": f"{doc_id}__section_summary__",
                }),
            )
        )

        for subsection_id, subsection in subsections.items():
            total_subsections += 1

            meta = safe_meta({
                **base_meta,
                "subsection_id": subsection_id,
                "official_subsection_id": subsection.get("official_subsection_id", ""),
                "heading": subsection.get("heading", ""),
                "ordinal": subsection.get("ordinal", ""),
                "chunk_type": "subsection_summary",
                "chunk_id": f"{doc_id}__{subsection_id}",
            })

            # Dense DB gets lightweight summary + keywords + IDs.
            child_docs.append(
                Document(
                    page_content=build_dense_subsection_page(parent, subsection),
                    metadata=meta,
                )
            )

            # Sparse BM25 gets the original paragraph text for lexical matching.
            sparse_docs.append(
                Document(
                    page_content=build_sparse_subsection_page(parent, subsection),
                    metadata={**meta, "chunk_type": "subsection_original_text"},
                )
            )

    print(f"Parent docs: {len(parent_registry)}")
    print(f"Subsection evidence units: {total_subsections}")
    print(f"Dense child docs to embed: {len(child_docs)}")
    print(f"Sparse BM25 docs: {len(sparse_docs)}")

    registry_path = Path(PARENT_STORE_DIR) / "registry.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(parent_registry, f, ensure_ascii=False, indent=2)

    section_index_path = Path(PARENT_STORE_DIR) / "section_index.json"
    with section_index_path.open("w", encoding="utf-8") as f:
        json.dump(section_index, f, ensure_ascii=False, indent=2)

    stats_path = Path(PARENT_STORE_DIR) / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parent_docs": len(parent_registry),
                "subsection_evidence_units": total_subsections,
                "dense_child_docs": len(child_docs),
                "sparse_bm25_docs": len(sparse_docs),
                "folder_names": {
                    "parent_store": PARENT_STORE_DIR,
                    "vector_store": VECTOR_STORE_DIR,
                    "sparse_store": SPARSE_STORE_DIR,
                },
                "note": "Subsection ids are synthetic paragraph ids because current raw OSHA JSON does not preserve official paragraph labels.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {registry_path}")
    print(f"Saved: {section_index_path}")
    print(f"Saved: {stats_path}")

    print("\n[4/6] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding model loaded.")

    print("\n[5/6] Building Chroma vector store...")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(child_docs), batch_size):
        end = min(start + batch_size, len(child_docs))
        vectorstore.add_documents(child_docs[start:end])
        print(f"Embedded child docs {start + 1}-{end} / {len(child_docs)}")

    print(f"Saved Chroma DB folder: {VECTOR_STORE_DIR}")

    print("\n[6/6] Building BM25 sparse index...")
    if args.skip_bm25:
        print("Skipping BM25 because --skip-bm25 was used.")
    elif PersistentBM25Retriever is None:
        print("BM25 package not available. Skipping BM25 index.")
    else:
        PersistentBM25Retriever.from_documents(
            documents=sparse_docs,
            save_dir=SPARSE_STORE_DIR,
        )
        print(f"Saved BM25 folder: {SPARSE_STORE_DIR}")

    print("\n=== SUCCESS: OSHA RAG INDEX CREATED ===")
    print("Responder will receive subsection evidence, not full OSHA parent documents.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== SCRIPT FAILED ===")
        traceback.print_exc()
        input("\nPress Enter to close...")
        raise
