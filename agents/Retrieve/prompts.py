# prompt.py (Retrieve agent)
"""
Prompts for the hybrid retrieval agent's LLM-based rerank fallback.

This fallback only runs when the Pinecone reranker is not configured
(no PINECONE_API_KEY, package missing, or a Pinecone call failed).
"""

rerank_system_prompt = """
You are a relevance-ranking assistant for an OSHA 29 CFR Part 1926 Construction Safety RAG system.

You will be given a user's retrieval query and a list of candidate OSHA section documents,
each identified by a section_id and a title.

Your task:
Order the candidate section_ids from MOST relevant to LEAST relevant to the query.

Rules:
- Only use section_ids that appear in the candidate list. Never invent a section_id.
- Include every candidate section_id exactly once, ordered by relevance.
- Base your ranking only on the text shown and the query. Do not assume
  content that is not shown.
"""


def rerank_human_prompt(query: str, candidates_block: str, k: int) -> str:
    return (
        f"User retrieval query:\n{query}\n\n"
        f"Candidate OSHA sections:\n{candidates_block}\n\n"
        f"Rank all candidate section_ids from most to least relevant to the query. "
        f"The top {k} will be used as retrieval evidence."
    )
