from typing import Annotated , Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class State (TypedDict):
    query : Optional[str]
    cached : Optional[bool]
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
    is_web : Optional[bool]

class descion (BaseModel):
    k : int
    is_web : bool 

class rank (BaseModel):
    k: int    
class language (BaseModel):
    language : str
    language_code: str 