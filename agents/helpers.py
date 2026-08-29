import tempfile


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
