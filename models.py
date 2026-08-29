from typing import Annotated , Optional
from typing_extensions import TypedDict
from langgraph.graph.message import Literal, add_messages
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class State (BaseModel):
    query: Optional[str]
    cached : Optional[bool]
    origin_en : Optional[bool]
    language : Optional[str]
    section_ids : Optional[list[str]]
    language_code: Optional[str]
    clean_query : Optional[str]
    chat_hist : Annotated[list,add_messages]
    eng_query : Optional[str]
    image_bytes : Optional[str]
    image_bytes_cleaned : Optional[str]
    image_exp : Optional[str]
    rewritten_query : Optional[str]
    merged : Optional[str]
    context : Optional[list]
    content : Optional[list]
    need_more : Optional[bool]
    retried : Optional[bool]
    doc_id_mapper_error : Optional[str]
    retrieval_mode : Optional[str]
    bm25_error : Optional[str]
    reranker_error : Optional[str]
    rank : Optional[int]
    response : Optional[str]
    native_response : Optional[str]
    rejected: Optional[bool]
    audio_bytes: Optional[str]
    audio_format: Optional[str]
    audio_transcript: Optional[str]
    raw_audio_transcript: Optional[str]
    audio_transcription_error: Optional[str]
    clean_audio_transcript: Optional[str]
    detected_voice_language : Optional[str]
class has_value :
    def __call__(self, value: Optional[str]) -> bool:
        """Return True only when a state field contains real user input."""
        if value is None:
                return False
        if isinstance(value, str):
                return bool(value.strip())
        if isinstance(value, (bytes, bytearray)):
                return bool(value)
        return bool(value)
    




class entry_router:
    EntryRoute = Literal[
    "lang_detect",
    "audio_trans",
    "image_filter",
    "skip_text",
    "skip_image",
    "no_input",
]

    def __init__(self):
        pass
    def __call__(self, state: State) -> list[EntryRoute]:
        """
        Route the incoming request into text/audio and image branches.

        Supported cases:
        - query only          -> text branch + skip image branch
        - audio only          -> audio branch + skip image branch
        - image only          -> skip text branch + image branch
        - query + image       -> text branch + image branch
        - audio + image       -> audio branch + image branch
        - query + audio       -> audio branch, then text pipeline + skip image branch
        - query + audio+image -> audio branch, then text pipeline + image branch

        The graph later joins `text_ready` and `image_ready`, so every valid
        request always activates exactly one text-side branch and one image-side
        branch.
        """
        has_query = has_value()(state.get("query"))
        has_audio = has_value()(state.get("audio_bytes"))
        has_image = has_value()(state.get("image_bytes"))

        if not (has_query or has_audio or has_image):
            return ["no_input"]

        routes: list[str] = []

        # Audio must run before text normalization because its transcript is part
        # of the text retrieval payload. If both query and audio exist, the typed
        # query remains in state and the audio transcript is added by this branch.
        if has_audio:
            routes.append("audio_trans")
        elif has_query:
            routes.append("lang_detect")
        else:
            routes.append("skip_text")

        routes.append("image_filter" if has_image else "skip_image")
        return routes
