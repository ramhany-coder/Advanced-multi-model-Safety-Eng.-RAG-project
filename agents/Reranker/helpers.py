from agents.helpers import clamp_text

# How much of each chunk's text the reranker LLM gets to see. This is only for
# the ranking decision itself -- the full, unclamped chunk is still what gets
# forwarded to the responser once selected.
CHUNK_PREVIEW_CHARS = 600


def format_chunks_for_prompt(chunks: list) -> str:
    """Render candidate evidence chunks as a numbered block the LLM can rank by index."""
    blocks = []
    for i, doc in enumerate(chunks):
        metadata = getattr(doc, "metadata", {}) or {}
        section_id = metadata.get("section_id", "unknown")
        title = metadata.get("title", "")
        excerpt = clamp_text(getattr(doc, "page_content", "") or "", CHUNK_PREVIEW_CHARS)
        blocks.append(f"[{i}] Section {section_id} - {title}\n{excerpt}")
    return "\n\n".join(blocks)


def select_top_chunks(chunks: list, ranked_indices: list, top_k: int) -> list:
    """
    Build the final top_k evidence list from the LLM's ranked indices.

    Deduplicates and drops out-of-range/non-int indices, then pads with any
    remaining chunks (in their original retrieval order) if the LLM's ranking
    didn't cover enough valid indices to reach top_k.
    """
    seen = set()
    ordered = []

    for idx in ranked_indices or []:
        if isinstance(idx, bool):
            continue
        if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            ordered.append(chunks[idx])
        if len(ordered) >= top_k:
            return ordered

    if len(ordered) < top_k:
        for i, doc in enumerate(chunks):
            if i in seen:
                continue
            ordered.append(doc)
            seen.add(i)
            if len(ordered) >= top_k:
                break

    return ordered
