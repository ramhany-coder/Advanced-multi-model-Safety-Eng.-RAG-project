import tempfile

# Shared across the prompt modules of multiple agents (Rewrite, Merger, ...)
# that describe/enforce the retrieval query length budget.
MAX_RETRIEVAL_QUERY_CHARS = 400

# Shared corpus-awareness blurb reused by every agent prompt that needs the
# LLM to know what the local retrieval corpus actually contains (ImageAnalysis,
# Responser, QueryTranslator).
LOCAL_OSHA_1926_CORPUS_SUMMARY = (
    "The local retrieval corpus contains OSHA 29 CFR Part 1926 construction safety "
    "regulation section documents. It includes about 374 OSHA 1926 sections with "
    "section_id, title, url, and full_text fields. Covered construction topics include "
    "general construction safety requirements, scaffolds, fall protection, PPE, ladders, "
    "stairways, excavations, trenching, cranes, derricks, hoists, aerial lifts, confined "
    "spaces in construction, electrical safety, toxic and hazardous substances, steel "
    "erection, demolition, concrete and masonry construction, fire protection, material "
    "handling, tools, welding and cutting, signs/signals/barricades, motor vehicles, "
    "mechanized equipment, rollover protection, underground construction, blasting, "
    "power transmission and distribution, and related OSHA 1926 construction standards."
)


def tempfile_creator(audio_bytes, audio_formate):
    try:
        suffix = audio_formate if audio_formate.startswith(".") else f".{audio_formate}"
        temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )
        temp_file.write(audio_bytes)
        temp_file.flush()
        temp_file.close()
        audio_path = temp_file.name
    except Exception as e :
        raise ValueError(f"Error during creating temp file for audio scripting {e}")
    return temp_file , audio_path

SUPPORTED_ROUTERS = ["ollama", "gpt", "gemini", "groq"]


def validate_router(router: str) -> str:
    if router not in SUPPORTED_ROUTERS:
        raise ValueError(
            f"Unsupported router '{router}'. "
            f"Available routers: {', '.join(SUPPORTED_ROUTERS)}"
        )
    return router

def combine_evidence(state) -> list:
    """
    Merge DB-matched sections (state['content'], from the doc-ID mapper) with
    whatever the hybrid retriever produced (state['context']), deduping by doc_id
    so the responser/ranker see the same evidence regardless of whether the
    retriever ran.
    """
    content = state.get("content") or []
    context = state.get("context") or []

    combined = []
    seen_doc_ids = set()
    for doc in list(content) + list(context):
        doc_id = getattr(doc, "metadata", {}).get("doc_id") if hasattr(doc, "metadata") else None
        if doc_id is not None:
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
        combined.append(doc)
    return combined


def format_context_for_prompt(
    context: list,
    max_docs: int | None = None,
    max_chars_per_doc: int | None = None,
) -> str:
    """
    Render retrieved evidence as compact "Section <id> - <title>" text blocks.

    Dumping the raw Document list (str(context)) wastes tokens on repeated
    class/metadata boilerplate and unclamped page_content for every item.
    max_docs/max_chars_per_doc let a caller (e.g. a tight-TPM router like
    Groq) shrink the payload without affecting callers that don't pass them.
    """
    docs = context[:max_docs] if max_docs else context
    if not docs:
        return "No retrieved context."

    blocks = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        section_id = metadata.get("section_id", "unknown")
        title = metadata.get("title", "")
        text = getattr(doc, "page_content", None)
        if text is None:
            text = str(doc)
        if max_chars_per_doc:
            text = clamp_text(text, max_chars_per_doc)
        blocks.append(f"Section {section_id} - {title}\n{text}")
    return "\n\n".join(blocks)


def clamp_text(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """
    Clamps a string to a maximum number of characters.
    
    :param text: The input string to truncate.
    :param max_chars: Maximum allowed characters.
    :param suffix: String to append if truncated (default: '...').
    :return: Clamped string.
    """
    if not text:
        return ""
    
    text = str(text)
    
    if len(text) <= max_chars:
        return text
    
    # Adjust length to accommodate suffix
    cutoff = max(0, max_chars - len(suffix))
    return text[:cutoff] + suffix
