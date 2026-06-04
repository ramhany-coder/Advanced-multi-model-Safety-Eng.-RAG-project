from typing import Annotated , Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class State (TypedDict):
    query : Optional[str]
    cached : Optional[bool]
    origin_en : Optional[bool]
    language : Optional[str]
    language_code: Optional[str]
    clean_query : Optional[str]
    chat_hist : Annotated[list,add_messages]
    eng_query : Optional[str]
    image_bytes : Optional[str]
    image_bytes_cleaned : Optional[str]
    image_exp : Optional[str]
    rewritten_query : Optional[str]
    merged : Optional[str]
    k : Optional[int]
    context : Optional[list]
    rank : Optional[int]
    response : Optional[str]
    native_response : Optional[str]
    is_web : Optional[bool]
    rejected: Optional[bool]
    audio_bytes: Optional[str]
    audio_format: Optional[str]
    audio_transcript: Optional[str]
    audio_transcription_error: Optional[str]
    clean_audio_transcript: Optional[str]
    detected_voice_language : Optional[str]
class descion (BaseModel):
    k : int
    is_web : bool 

class rank (BaseModel):
    k: int    
