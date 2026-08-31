"""Request/response payloads for the agent-testing API.

Every pipeline agent reads and writes a subset of the same shared state (see
models.State), so a single generic payload -- rather than one bespoke schema
per agent -- is used for every per-agent endpoint. All fields are optional:
send only the ones the target agent actually reads.
"""

from typing import Any, List, Optional

from pydantic import BaseModel


class StatePayload(BaseModel):
    query: Optional[str] = None
    cached: Optional[bool] = None
    origin_en: Optional[bool] = None
    language: Optional[str] = None
    language_code: Optional[str] = None
    clean_query: Optional[str] = None
    chat_hist: Optional[List[Any]] = None
    eng_query: Optional[str] = None
    image_bytes: Optional[str] = None
    image_bytes_cleaned: Optional[str] = None
    image_exp: Optional[str] = None
    rewritten_query: Optional[str] = None
    merged: Optional[str] = None
    content: Optional[List[Any]] = None
    retried: Optional[bool] = None
    rank: Optional[int] = None
    response: Optional[str] = None
    native_response: Optional[str] = None
    rejected: Optional[bool] = None
    audio_bytes: Optional[str] = None
    audio_format: Optional[str] = None
    audio_transcript: Optional[str] = None
    raw_audio_transcript: Optional[str] = None
    clean_audio_transcript: Optional[str] = None
    detected_voice_language: Optional[str] = None
    k: Optional[int] = None


class PipelineRequest(BaseModel):
    query: Optional[str] = None
    image_bytes: Optional[str] = None
    audio_bytes: Optional[str] = None
    audio_format: Optional[str] = None
    chat_hist: List[Any] = []
