from typing import Annotated , Optional
from typing_extensions import TypedDict
from langgraph.graph.message import Literal, add_messages
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class State (BaseModel):
    query: Optional[str] = None
    cached : Optional[bool] = None
    origin_en : Optional[bool] = None
    language : Optional[str] = None
    language_code: Optional[str] = None
    clean_query : Optional[str] = None
    chat_hist : Annotated[list,add_messages] = []
    eng_query : Optional[str] = None
    image_bytes : Optional[str] = None
    image_bytes_cleaned : Optional[str] = None
    image_exp : Optional[str] = None
    rewritten_query : Optional[str] = None
    merged : Optional[str] = None
    sub_queries : list[str] = []
    decomposer_error : Optional[str] = None
    content : Optional[list] = None
    # One retrieval hit list per sub-query, kept separate (not unioned) so the
    # reranker node's RRF fusion (agents/Retrieve/fusion.py) can weigh a chunk
    # by how many sub-queries surfaced it, not just merge everything together.
    ranked_lists : Optional[list] = None
    retried : Optional[bool] = None
    retrieval_mode : Optional[str] = None
    bm25_error : Optional[str] = None
    reranker_error : Optional[str] = None
    rank : Optional[int] = None
    cache_verdict : Optional[str] = None
    response : Optional[str] = None
    native_response : Optional[str] = None
    rejected: Optional[bool] = None
    audio_bytes: Optional[str] = None
    audio_format: Optional[str] = None
    audio_transcript: Optional[str] = None
    raw_audio_transcript: Optional[str] = None
    audio_transcription_error: Optional[str] = None
    clean_audio_transcript: Optional[str] = None
    detected_voice_language : Optional[str] = None

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

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
        has_query = has_value()(state.query)
        has_audio = has_value()(state.audio_bytes)
        has_image = has_value()(state.image_bytes)

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
