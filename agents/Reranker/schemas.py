from pydantic import BaseModel


class RerankSelection(BaseModel):
    """Structured output for ranking candidate evidence chunks by relevance to the query."""
    ranked_chunk_indices: list[int]
