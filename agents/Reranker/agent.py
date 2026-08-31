from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from agents.llm.fallback import FallBack
from agents.helpers import combine_evidence
from agents.Reranker.helpers import format_chunks_for_prompt, select_top_chunks
from agents.Reranker.prompts import reranker_human_prompt, reranker_system_prompt
from agents.Reranker.schemas import RerankSelection
from agents.Retrieve.fusion import rrf_fuse

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-safeguard-20b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

# RRF fuses to a slightly wider pool than the final target so the (optional)
# LLM path below has more than settings.RETRIEVAL_TOP_K candidates to choose
# its top-k from. The original query is weighted up -- it has repeatedly
# found paragraphs no decomposition thought to search for.
RRF_CANDIDATE_POOL = 12
RRF_QUERY_WEIGHTS = {"0": 1.5}

reranker_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def reranker_agent(state) -> dict:
    """
    Trim the retrieved evidence down to settings.RETRIEVAL_TOP_K chunks most
    relevant to the merged query before the responser sees it.

    Deterministic RRF fusion (agents/Retrieve/fusion.py) is the default: it
    fuses each sub-query's own ranked hit list by rank position -- comparable
    across sub-queries, unlike raw similarity scores -- weighted by
    chunk_corpus.py's retrieval_weight tag so a chunk that only shares
    vocabulary with the query doesn't outrank the paragraph that actually
    governs it. Set settings.USE_LLM_RERANKER to route the fused candidate
    pool through the LLM path instead; left in place for A/B comparison
    against the golden set, not because it currently outperforms RRF.
    """
    query = state.get("merged") or ""
    ranked_lists = state.get("ranked_lists")

    if ranked_lists:
        docs = rrf_fuse(ranked_lists, top_k=RRF_CANDIDATE_POOL, weights=RRF_QUERY_WEIGHTS)
    else:
        # No per-query lists to fuse (e.g. a cache-reasoner path that never
        # ran the retriever this pass) -- fall back to whatever
        # combine_evidence can dedupe from state['content'].
        docs = combine_evidence(state)

    if settings.USE_LLM_RERANKER and len(docs) > settings.RETRIEVAL_TOP_K:
        messages = [
            SystemMessage(content=reranker_system_prompt(settings.RETRIEVAL_TOP_K)),
            HumanMessage(
                content=reranker_human_prompt(
                    query=query,
                    chunks_block=format_chunks_for_prompt(docs),
                    total_chunks=len(docs),
                    top_k=settings.RETRIEVAL_TOP_K,
                )
            ),
        ]
        try:
            result = reranker_llm.constrained_invoke(
                messages, FALLBACK_ORDER, constraine_model=RerankSelection
            )
            ranked_indices = result.get("ranked_chunk_indices") or []
            selected = select_top_chunks(docs, ranked_indices, settings.RETRIEVAL_TOP_K)
            return {"content": selected}
        except Exception as e:
            return {
                "content": docs[: settings.RETRIEVAL_TOP_K],
                "reranker_error": str(e),
            }

    docs = docs[: settings.RETRIEVAL_TOP_K]  # already RRF-ordered and weighted
    return {"content": docs}
