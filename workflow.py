from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import START, END, StateGraph
from langsmith import traceable

from models import State
from agents.agents import *
from agents.Retrieve.agent import hyb_retriver_agent
from agents.DocIdMapper.agent import doc_id_mapper_agent






def detect_language_from_available_text(state: State) -> dict[str, Any]:
    """
    Detect the user-facing language from typed text and/or audio transcript.

    The original agent only checked `query`, which makes audio-only Arabic
    requests default to English. This wrapper keeps the workflow robust without
    modifying agents.py.
    """
    text_parts = [
        state.get("query") or "",
        state.get("audio_transcript") or "",
        state.get("raw_audio_transcript") or "",
    ]
    text = "\n".join(part.strip() for part in text_parts if part and part.strip())

    if not text:
        return {
            "language": "English",
            "language_code": "en",
            "origin_en": True,
        }

    detected = language_detector.detect_language_of(text)

    if detected is None:
        return {
            "language": "English",
            "language_code": "en",
            "origin_en": True,
        }

    lang_name = detected.name.capitalize()
    lang_code = detected.iso_code_639_1.name.lower()

    return {
        "language": lang_name,
        "language_code": lang_code,
        "origin_en": lang_code == "en",
    }


def skip_text_agent(state: State) -> dict[str, Any]:
    """Mark the text branch as intentionally empty."""
    return {
        "clean_query": state.get("clean_query") or "",
        "eng_query": state.get("eng_query") or "",
        "rewritten_query": state.get("rewritten_query") or "",
    }


def skip_image_agent(state: State) -> dict[str, Any]:
    """Mark the image branch as intentionally empty."""
    return {
        "image_bytes_cleaned": state.get("image_bytes_cleaned"),
        "image_exp": state.get("image_exp") or "",
    }


def passthrough_agent(state: State) -> dict[str, Any]:
    """Used as a join marker. It intentionally does not modify state."""
    return {}


def no_input_agent(state: State) -> dict[str, Any]:
    """Return a safe response when the caller provided no usable input."""
    return {
        "language": "English",
        "language_code": "en",
        "origin_en": True,
        "response": (
            "Please provide a text query, an image, or an audio file so I can "
            "perform the OSHA safety-compliance analysis."
        ),
        "rejected": True,
    }


def safe_merging_agent(state: State) -> dict[str, Any]:
    """
    Merge text/audio and image evidence safely.

    The original merger could receive None for text-only or image-only requests.
    This wrapper normalizes missing branches to empty strings.
    """
    normalized_state = dict(state)
    normalized_state["rewritten_query"] = state.get("rewritten_query") or ""
    normalized_state["image_exp"] = state.get("image_exp") or ""

    if not normalized_state["rewritten_query"] and not normalized_state["image_exp"]:
        return {
            "merged": (
                "No usable OSHA safety query, audio transcript, or image analysis "
                "was provided."
            )
        }

    return merging_agent(normalized_state) or {}


def safe_caching_agent(state: State) -> dict[str, Any]:
    """
    Ensure the cache node always returns a dict.

    The original caching_agent may return None after writing to cache, which can
    break LangGraph state updates.
    """
    return caching_agent(state) or {}


def cache_router(state: State) -> Literal["jump", "continue"]:
    return "jump" if bool(state.get("cached")) else "continue"


def retrieval_necessity_router(state: State) -> Literal["need_retrieval", "skip_retrieval"]:
    """
    The hybrid retriever is only needed when the doc-ID mapper couldn't
    confidently pin the query to enough OSHA sections on its own.
    """
    section_ids = state.get("section_ids") or []
    need_more = bool(state.get("need_more"))

    return "need_retrieval" if (len(section_ids) < 3 or need_more) else "skip_retrieval"


def rank_router(state: State) -> Literal["accepted", "rejected"]:
    try:
        rank_value = int(state.get("rank") or 0)
    except (TypeError, ValueError):
        rank_value = 0

    return "accepted" if rank_value >= 7 else "rejected"


class Workflow:
    def __init__(self) -> None:
        self.responser = responser_agent
        self.audio_trans = audio_transcription_agent
        self.lang_detector = detect_language_from_available_text
        self.user_query_trans = user_query_translator
        self.query_filter = query_pii_agent
        self.image_filter = image_pii_agent
        self.merger = safe_merging_agent
        self.is_cache = check_cache_agent
        self.k_getter = k_getter_agent
        self.doc_id_mapper = doc_id_mapper_agent
        self.retriever = hyb_retriver_agent
        self.rewriter = rewrite_agent
        self.image = image_exp_agent
        self.caching_agent = safe_caching_agent
        self.ranker = ranker_agent
        self.response_trans = response_translator
        self.rejection_response = rejection_response_agent

    def compile(self):
        graph = StateGraph(State)

        # Entry / branch-control nodes
        graph.add_node("skip_text", skip_text_agent)
        graph.add_node("skip_image", skip_image_agent)
        graph.add_node("text_ready", passthrough_agent)
        graph.add_node("image_ready", passthrough_agent)
        graph.add_node("no_input", no_input_agent)

        # Main pipeline nodes
        graph.add_node("audio_trans", self.audio_trans)
        graph.add_node("lang_detect", self.lang_detector)
        graph.add_node("query_filter", self.query_filter)
        graph.add_node("user_trans", self.user_query_trans)
        graph.add_node("query_rewriter", self.rewriter)
        graph.add_node("image_filter", self.image_filter)
        graph.add_node("image", self.image)
        graph.add_node("merger", self.merger)
        graph.add_node("cache_check", self.is_cache)
        graph.add_node("k_getter", self.k_getter)
        graph.add_node("doc_id_mapper", self.doc_id_mapper)
        graph.add_node("retriever", self.retriever)
        graph.add_node("responser", self.responser)
        graph.add_node("ranker", self.ranker)
        graph.add_node("caching", self.caching_agent)
        graph.add_node("rejection_response", self.rejection_response)
        graph.add_node("response_trans", self.response_trans)

        graph.add_conditional_edges(
            START,
            entry_router,
            {
                "lang_detect": "lang_detect",
                "audio_trans": "audio_trans",
                "image_filter": "image_filter",
                "skip_text": "skip_text",
                "skip_image": "skip_image",
                "no_input": "no_input",
            },
        )

        # No-input path
        graph.add_edge("no_input", "response_trans")

        # Text/audio branch
        graph.add_edge("audio_trans", "lang_detect")
        graph.add_edge("lang_detect", "query_filter")
        graph.add_edge("query_filter", "user_trans")
        graph.add_edge("user_trans", "query_rewriter")
        graph.add_edge("query_rewriter", "text_ready")
        graph.add_edge("skip_text", "text_ready")

        # Image branch
        graph.add_edge("image_filter", "image")
        graph.add_edge("image", "image_ready")
        graph.add_edge("skip_image", "image_ready")

        # Wait until both selected branches are ready before merging.
        graph.add_edge(["text_ready", "image_ready"], "merger")

        # RAG / response path
        graph.add_edge("merger", "cache_check")
        graph.add_conditional_edges(
            "cache_check",
            cache_router,
            {
                "jump": "response_trans",
                "continue": "k_getter",
            },
        )
        graph.add_edge("k_getter", "doc_id_mapper")
        graph.add_conditional_edges(
            "doc_id_mapper",
            retrieval_necessity_router,
            {
                "need_retrieval": "retriever",
                "skip_retrieval": "responser",
            },
        )
        graph.add_edge("retriever", "responser")
        graph.add_edge("responser", "ranker")
        graph.add_conditional_edges(
            "ranker",
            rank_router,
            {
                "accepted": "caching",
                "rejected": "rejection_response",
            },
        )
        graph.add_edge("rejection_response", "response_trans")
        graph.add_edge("caching", "response_trans")
        graph.add_edge("response_trans", END)

        return graph.compile()

    @traceable
    def run(self, initial_state: State) -> dict[str, Any]:
        graph = self.compile()
        return graph.invoke(initial_state)


workflow = Workflow

client = workflow()

print(client.run(State(query="Safety requirements for arc welding")))
