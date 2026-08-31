from pydantic import BaseModel, Field


class QueryDecomposition(BaseModel):
    """Structured output for splitting a merged query into targeted retrieval phrases."""
    sub_queries: list[str] = Field(
        description="3-6 self-contained retrieval phrases, most important first. "
                    "The first element is the original merged query, unchanged."
    )
