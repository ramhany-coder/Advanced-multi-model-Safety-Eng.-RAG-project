"""Runs the full compiled LangGraph pipeline from workflow.py for the API,
timing every substantive agent node without touching the graph topology
defined there.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from workflow import Workflow

logger = logging.getLogger("pipeline")

# Instance attributes on Workflow that hold real agent work (see workflow.py's
# Workflow.__init__ / compile). Control-flow glue nodes (skip_*, passthroughs,
# no_input, mark_retry) are intentionally not timed.
_TIMED_NODE_ATTRS = [
    "audio_trans",
    "lang_detector",
    "query_filter",
    "user_query_trans",
    "image_filter",
    "image",
    "merger",
    "is_cache",
    "doc_id_mapper",
    "retriever",
    "reranker",
    "responser",
    "ranker",
    "caching_agent",
    "rejection_response",
    "response_trans",
]


class PipelineStageError(Exception):
    """Raised when a graph node fails. Carries the failing stage's name, how
    long it ran before failing, the latencies of every stage that completed
    before it, and the state going into the failing stage."""

    def __init__(
        self,
        stage: str,
        elapsed: float,
        stage_timings: Dict[str, float],
        state: Dict[str, Any],
        original: Exception,
    ) -> None:
        self.stage = stage
        self.elapsed = elapsed
        self.stage_timings = stage_timings
        self.state = state
        self.original = original
        super().__init__(f"stage '{stage}' failed after {elapsed:.2f}s: {original}")


def _timed(name: str, fn: Callable, timings: Dict[str, float]) -> Callable:
    def wrapper(state):
        start = time.perf_counter()
        try:
            result = fn(state)
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(
                "[pipeline] stage '%s' failed after %.2fs: %s", name, elapsed, e
            )
            raise PipelineStageError(name, elapsed, dict(timings), dict(state), e) from e

        elapsed = time.perf_counter() - start
        timings[name] = round(elapsed, 3)
        logger.info("[pipeline] stage '%s' completed in %.2fs", name, elapsed)
        return result

    return wrapper


def _build_timed_workflow(timings: Dict[str, float]) -> Workflow:
    """A fresh Workflow instance per run, with each agent attribute wrapped so
    its latency lands in `timings` (kept outside graph state, since the
    pydantic State schema in models.py has no stage_timings field)."""
    wf = Workflow()
    for attr in _TIMED_NODE_ATTRS:
        setattr(wf, attr, _timed(attr, getattr(wf, attr), timings))
    return wf


def build_initial_state(
    query: Optional[str] = None,
    image_bytes: Optional[str] = None,
    audio_bytes: Optional[str] = None,
    audio_format: Optional[str] = None,
    chat_hist: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build a full initial state dict covering every models.State field, so
    the graph never has to fall back on a field being silently absent."""
    return {
        "query": query,
        "cached": None,
        "origin_en": None,
        "language": None,
        "section_ids": None,
        "language_code": None,
        "clean_query": None,
        "chat_hist": chat_hist or [],
        "eng_query": None,
        "image_bytes": image_bytes,
        "image_bytes_cleaned": None,
        "image_exp": None,
        "rewritten_query": None,
        "merged": None,
        "context": None,
        "content": None,
        "need_more": None,
        "retried": None,
        "doc_id_mapper_error": None,
        "retrieval_mode": None,
        "bm25_error": None,
        "reranker_error": None,
        "rank": None,
        "response": None,
        "native_response": None,
        "rejected": None,
        "audio_bytes": audio_bytes,
        "audio_format": audio_format,
        "audio_transcript": None,
        "raw_audio_transcript": None,
        "audio_transcription_error": None,
        "clean_audio_transcript": None,
        "detected_voice_language": None,
    }


def run_pipeline(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """Runs translator/PII/retrieval/ranking end to end via the same compiled
    graph as workflow.Workflow, returning the final state plus per-stage and
    total latency. Raises PipelineStageError if any stage fails."""
    timings: Dict[str, float] = {}
    graph = _build_timed_workflow(timings).compile()

    start = time.perf_counter()
    try:
        final_state = graph.invoke(initial_state)
    except PipelineStageError as e:
        total_elapsed = time.perf_counter() - start
        logger.error(
            "[pipeline] aborted after %.2fs total - failed stage: '%s'",
            total_elapsed,
            e.stage,
        )
        raise

    total_elapsed = time.perf_counter() - start
    logger.info(
        "[pipeline] completed in %.2fs - stage_timings=%s", total_elapsed, timings
    )

    return {
        **dict(final_state),
        "stage_timings": timings,
        "total_latency_seconds": round(total_elapsed, 3),
    }
