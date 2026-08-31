from gptcache import cache
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.Cache.prompts import cache_reasoner_human_prompt, cache_reasoner_system_prompt
from agents.Cache.schemas import CacheAlignmentVerdict

cache.init(pre_embedding_func=get_prompt)

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-safeguard-20b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

cache_reasoner_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def check_cache_agent(state) -> dict:
    query = state.get('merged')
    result = cache_get(query)
    if result:
        return {'cached': True, "response": result}
    else:
        return {"cached": False}


def cache_reasoner_agent(state) -> dict:
    """
    Optional guard on a cache hit: the cache layer can match the current
    query to a past query that only looks similar, not identical, so a
    reused response can miss what the user actually asked. This asks an LLM
    to compare the current query against the cached response and decide
    whether to reuse it as-is, rewrite it to fit, or discard the hit and let
    the query fall through to normal retrieval.

    Only reached when settings.ENABLE_CACHE_REASONING is True (see
    workflow.cache_router) - with it False, a cache hit is trusted as before.
    """
    query = state.get("merged") or state.get("eng_query") or ""
    cached_response = state.get("response") or ""

    if not cached_response:
        return {"cache_verdict": "recompute", "cached": False}

    messages = [
        SystemMessage(content=cache_reasoner_system_prompt),
        HumanMessage(content=cache_reasoner_human_prompt(query, cached_response)),
    ]

    try:
        result = cache_reasoner_llm.constrained_invoke(
            messages, fallback_order=FALLBACK_ORDER, constraine_model=CacheAlignmentVerdict
        )
        verdict = result.get("verdict")

        if verdict == "recompute":
            return {"cache_verdict": "recompute", "cached": False}

        if verdict == "refine" and result.get("refined_response"):
            return {"cache_verdict": "refine", "response": result["refined_response"]}

        return {"cache_verdict": "reuse"}

    except Exception:
        # Fail conservatively: don't serve a possibly-misaligned cached
        # answer just because the auditor call itself failed - fall through
        # to a fresh retrieval pass instead.
        return {"cache_verdict": "recompute", "cached": False}


def caching_agent(state) -> dict:
    caching_stat = state.get('cached')
    if not caching_stat:
        query = state.get('merged')
        response = state.get('response')
        if response and query:
            cache_put(query, response)
