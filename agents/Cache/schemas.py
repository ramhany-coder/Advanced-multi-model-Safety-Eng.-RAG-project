from typing import Literal, Optional
from pydantic import BaseModel


class CacheAlignmentVerdict(BaseModel):
    """Structured verdict for whether a cache hit still fits the live query.

    reuse      Cached response answers the current query as-is.
    refine     Same topic/section, but wording needs adjusting to the current
               query's phrasing/detail - `refined_response` must be filled in.
    recompute  Cached response does not actually address the current query -
               treat this as a cache miss and run fresh retrieval.
    """
    verdict: Literal["reuse", "refine", "recompute"]
    refined_response: Optional[str] = None
