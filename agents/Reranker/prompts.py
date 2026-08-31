# -*- coding: utf-8 -*-
"""
Prompts for the pre-response evidence reranking agent.

Runs right before the responser, after the doc-ID mapper and/or hybrid
retriever have produced the candidate pool. Its only job is to pick the chunks
that most directly answer the merged query.

WHAT CHANGED AND WHY IT IS FASTER
---------------------------------
1. It no longer returns a total ordering. The old prompt required "every
   candidate index exactly once, ordered most to least relevant" - so with 20
   candidates the model had to reason about and emit all 20, including the
   ranking of chunks nobody would ever read. It now returns at most top_k
   indices and stops. Output tokens drop by roughly the ratio of candidates to
   top_k, and so does the reasoning that produces them.

2. The chunks_block is capped per candidate. Chunks average ~1,200 characters;
   twenty of them is ~24,000 characters of input on every call. The reranker
   does not need full text to judge relevance - a citation, a title and the
   opening ~280 characters is enough. Use build_chunks_block() below.

3. The call is skipped entirely when there is nothing to cut. See
   should_rerank().

4. Run it with groq_reasoning_effort="low". Selection is a judgment call, not
   a chain of deduction.

COMPATIBILITY
-------------
The output field is still named `ranked_chunk_indices`, so downstream code that
does `indices[:top_k]` keeps working unchanged. The only difference is that the
list is now already at most top_k long.

Update the Pydantic schema's description to match:

    class ChunkRanking(BaseModel):
        ranked_chunk_indices: list[int] = Field(
            description="At most top_k candidate indices, best first. "
                        "Not a total ordering - unselected indices are omitted."
        )
"""

MAX_CHUNK_PREVIEW = 280


reranker_system_prompt_template = """
You are an Evidence Selection Engine for an OSHA 29 CFR Part 1926 construction
safety RAG pipeline.

You receive a retrieval query and a numbered list of candidate passages already
pulled from the corpus. Select the **{top_k} most useful** passages for
answering the query, best first. Do not rank the rest - omit them.

## HOW TO JUDGE

**Judge the passage text, not the section title.** Each candidate shows the
title of the whole section it came from. That section covers many subjects and
its title is often unrelated to the passage in front of you. OSHA places duties
where the drafters put them: a duty about vehicles sits inside the excavation
subpart, a duty about signage inside the demolition subpart.

    title    : 1926.651 - Specific excavation requirements
    text     : "(e) Exposure to falling loads ... Employees shall be required
                to stand away from any vehicle being loaded or unloaded ..."
    query    : safety procedures for unloading a truck
    -> SELECT. The passage states the governing duty. The title is irrelevant.

The reverse holds too. A passage from "1926.602 - Material handling equipment"
about overhead guards on high-lift rider industrial trucks does not answer a
question about unloading a dump truck. Match on the duty described, never on
words shared with the heading.

**Prefer coverage over repetition.** A real OSHA answer is assembled from
several duties. Eight passages about the same requirement are worth less than
five that each cover a different one. When two passages state substantially the
same duty, keep the more specific and spend the slot elsewhere.

**Prefer operative text.** A passage that states a requirement - "shall",
"must", a number, a distance, a condition - outranks one that defines scope,
lists definitions, or describes a training program.

**Select fewer than {top_k} when the pool is thin.** Padding the list with
passages that bear no relation to the query makes the answer worse, not longer.
It is correct to return 3 when only 3 are relevant.

## OUTPUT

Return `ranked_chunk_indices`: at most {top_k} indices, most useful first.
Only indices shown to you. No duplicates. No explanation, no preamble - the
indices are the entire output.

## EXAMPLE

query: "Does a worker need fall protection while standing on this scaffold?"
candidates:
  0: 1926.451 - scaffold guardrail and platform requirements
  1: 1926.501 - duty to have fall protection
  2: 1926.652 - excavation protective systems
  3: 1926.451 - scaffold access and egress (ladders)
  4: 1926.451 - scaffold capacity and load ratings

ranked_chunk_indices: [1, 0, 4]

  1 and 0 state the duty directly. 4 is a different scaffold duty that bears on
  safe use, so it earns the third slot over 3, which is about access rather
  than fall protection. 2 is unrelated and is omitted rather than ranked - the
  list stops when the useful passages run out.
"""


def reranker_system_prompt(top_k: int) -> str:
    return reranker_system_prompt_template.format(top_k=top_k)


def reranker_human_prompt(query: str, chunks_block: str, total_chunks: int,
                          top_k: int) -> str:
    return (
        f"Query:\n{query}\n\n"
        f"Candidates (indices 0-{max(total_chunks - 1, 0)}):\n{chunks_block}\n\n"
        f"Return the {top_k} most useful indices, best first. "
        f"Fewer if fewer are relevant. Indices only."
    )


# ---------------------------------------------------------------------------
# Helpers for the agent module
# ---------------------------------------------------------------------------

def build_chunks_block(docs, preview: int = MAX_CHUNK_PREVIEW) -> str:
    """
    Render candidates compactly.

    Full chunk text is the reranker's main latency cost and it is not needed to
    judge relevance. Citation + title + opening sentences carry the signal.
    """
    lines = []
    for i, doc in enumerate(docs):
        md = getattr(doc, "metadata", None) or {}
        text = (getattr(doc, "page_content", "") or "").strip()

        # Strip the "Title: ...\n\nText:\n" header the corpus prepends, so the
        # preview budget is spent on regulation text rather than a repeated
        # header we are already showing on its own line.
        marker = "\nText:\n"
        if marker in text:
            text = text.split(marker, 1)[1]
        text = " ".join(text.split())

        if len(text) > preview:
            text = text[:preview].rsplit(" ", 1)[0] + " …"

        cite = md.get("citation") or md.get("section_id") or ""
        title = md.get("title") or ""
        lines.append(f"{i}: [{cite}] {title}\n   {text}")
    return "\n\n".join(lines)


def should_rerank(docs, top_k: int) -> bool:
    """
    Skip the LLM when it has nothing to cut.

    The old threshold was a hardcoded 5 while top_k was configurable, so the
    two could disagree. Tie it to top_k instead.
    """
    return len(docs) > top_k


def apply_ranking(docs, ranked_indices, top_k: int):
    """
    Map the model's indices back to documents, defensively.

    An out-of-range or duplicated index is dropped rather than raising. If the
    model returns nothing usable, fall back to the retriever's own order -
    those documents were already scored against this query, so that ordering is
    a sane default and never worse than an arbitrary slice.
    """
    seen, out = set(), []
    for idx in ranked_indices or []:
        if not isinstance(idx, int) or idx < 0 or idx >= len(docs) or idx in seen:
            continue
        seen.add(idx)
        out.append(docs[idx])
        if len(out) >= top_k:
            break
    return out or list(docs)[:top_k]