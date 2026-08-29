from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.helpers import combine_evidence
from agents.Reranker.helpers import format_chunks_for_prompt, select_top_chunks
from agents.Reranker.prompts import reranker_human_prompt, reranker_system_prompt
from agents.Reranker.schemas import RerankSelection

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "llama-3.1-8b-instant"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

# Only rerank when there's actually more evidence than the responser needs.
RERANK_THRESHOLD = 5
RERANK_TOP_K = 5

reranker_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def reranker_agent(state) -> dict:
    """
    Trim the retrieved evidence down to the RERANK_TOP_K chunks most relevant
    to the merged query before the responser sees it.

    Runs unconditionally right before the responser, regardless of whether the
    evidence came from the doc-ID mapper alone (skip_retrieval path), the
    hybrid retriever (need_retrieval path), or both -- combine_evidence()
    already merges and dedupes state['content'] + state['context'], so this
    agent doesn't need to know which path produced them.

    It's a no-op (aside from consolidating into 'context') when there are
    RERANK_THRESHOLD or fewer candidate chunks to begin with.
    """
    query = state.get("merged") or ""
    combined = combine_evidence(state)

    if len(combined) <= RERANK_THRESHOLD:
        # Nothing to trim, but still consolidate into 'context' so downstream
        # nodes have a single, consistent source of evidence to read from.
        return {"context": combined, "content": []}

    messages = [
        SystemMessage(content=reranker_system_prompt(RERANK_TOP_K)),
        HumanMessage(
            content=reranker_human_prompt(
                query=query,
                chunks_block=format_chunks_for_prompt(combined),
                total_chunks=len(combined),
                top_k=RERANK_TOP_K,
            )
        ),
    ]

    try:
        result = reranker_llm.constrained_invoke(
            messages, FALLBACK_ORDER, constraine_model=RerankSelection
        )
        ranked_indices = result.get("ranked_chunk_indices") or []
        selected = select_top_chunks(combined, ranked_indices, RERANK_TOP_K)
        return {"context": selected, "content": []}
    except Exception as e:
        return {
            "context": combined[:RERANK_TOP_K],
            "content": [],
            "reranker_error": str(e),
        }
