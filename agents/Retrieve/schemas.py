from pydantic import BaseModel


class RerankSelection(BaseModel):
    """Structured output for the LLM rerank fallback used when Pinecone is unavailable."""
    ranked_section_ids: list[str]
