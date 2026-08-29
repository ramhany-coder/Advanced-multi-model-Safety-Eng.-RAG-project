from pydantic import BaseModel


class RankScore(BaseModel):
    """Structured output for the QA ranker's 0-10 answer-quality score."""
    k: int
