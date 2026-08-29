from pydantic import BaseModel


class DocIdMapping(BaseModel):
    """Structured output for mapping a retrieval query to known OSHA section IDs."""
    section_ids: list[str]
    need_more: bool
