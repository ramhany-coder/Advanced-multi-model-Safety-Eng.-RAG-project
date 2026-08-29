import logging
import time
from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException

from agents.Audio.agent import audio_transcription_agent
from agents.PII.agent import query_pii_agent, image_pii_agent
from agents.QueryTranslator.agent import user_query_translator
from agents.Rewrite.agent import rewrite_agent
from agents.ImageAnalysis.agent import image_exp_agent
from agents.Cache.agent import check_cache_agent, caching_agent
from agents.DocIdMapper.agent import doc_id_mapper_agent
from agents.Retrieve.agent import hyb_retriver_agent
from agents.Reranker.agent import reranker_agent
from agents.Responser.agent import responser_agent
from agents.Ranker.agent import ranker_agent, rejection_response_agent
from agents.ResponseTranslator.agent import response_translator
from workflow import detect_language_from_available_text, safe_merging_agent

from api.schemas import StatePayload, PipelineRequest
from api.workflow import build_initial_state, run_pipeline, PipelineStageError

router = APIRouter()
logger = logging.getLogger("pipeline")

# Every entry runs in isolation against whatever fields of StatePayload the
# caller sends -- none of these depend on the others having run first.
STAGE_AGENTS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "audio-transcription": audio_transcription_agent,
    "language-detector": detect_language_from_available_text,
    "query-pii": query_pii_agent,
    "image-pii": image_pii_agent,
    "user-query-translator": user_query_translator,
    "rewrite": rewrite_agent,
    "image-analysis": image_exp_agent,
    "merger": safe_merging_agent,
    "cache-check": check_cache_agent,
    "cache-write": lambda state: caching_agent(state) or {},
    "doc-id-mapper": doc_id_mapper_agent,
    "retriever": hyb_retriver_agent,
    "reranker": reranker_agent,
    "responser": responser_agent,
    "ranker": ranker_agent,
    "rejection-response": rejection_response_agent,
    "response-translator": response_translator,
}


def _run_stage(name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Time a single-stage call; log and raise a 500 with the stage name + latency on failure."""
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("[api] stage '%s' failed after %.2fs: %s", name, elapsed, e)
        raise HTTPException(
            status_code=500,
            detail={
                "failed_stage": name,
                "stage_latency_seconds": round(elapsed, 3),
                "error": str(e),
            },
        )

    elapsed = time.perf_counter() - start
    logger.info("[api] stage '%s' completed in %.2fs", name, elapsed)
    result = dict(result or {})
    result["latency_seconds"] = round(elapsed, 3)
    return result


def _make_stage_handler(name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    def handler(payload: StatePayload) -> Dict[str, Any]:
        return _run_stage(name, lambda: fn(payload.model_dump()))

    handler.__name__ = f"run_{name.replace('-', '_')}"
    return handler


@router.get("/agents")
def list_agents():
    """List the agent stages that can be exercised individually via /agents/{name}."""
    return {"agents": sorted(STAGE_AGENTS.keys())}


for _name, _fn in STAGE_AGENTS.items():
    router.add_api_route(
        f"/agents/{_name}",
        _make_stage_handler(_name, _fn),
        methods=["POST"],
        name=_name,
        summary=f"Run the '{_name}' agent in isolation",
        tags=["agents"],
    )


# ---- full pipeline ----


@router.post("/pipeline/run", tags=["pipeline"])
def run_full_pipeline(payload: PipelineRequest):
    initial_state = build_initial_state(
        query=payload.query,
        image_bytes=payload.image_bytes,
        audio_bytes=payload.audio_bytes,
        audio_format=payload.audio_format,
        chat_hist=payload.chat_hist,
    )
    try:
        return run_pipeline(initial_state)
    except PipelineStageError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "failed_stage": e.stage,
                "stage_latency_seconds": round(e.elapsed, 3),
                "completed_stage_timings": e.stage_timings,
                "state": e.state,
                "error": str(e.original),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
