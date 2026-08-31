from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.QueryDecomposer.prompts import (
    query_decomposer_human_prompt,
    query_decomposer_system_prompt,
)
from agents.QueryDecomposer.schemas import QueryDecomposition

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-safeguard-20b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

MAX_SUB_QUERIES = 6

decomposer_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def _normalize_sub_queries(raw_queries: list, merged: str) -> list[str]:
    """
    Dedupe/clean the model's proposed sub-queries and force `merged` into
    position 0 regardless of what the model returned -- the model is a hint,
    not the source of truth for ordering. Caps at MAX_SUB_QUERIES.
    """
    merged = merged or ""
    seen = {merged.strip().lower()}
    cleaned = [merged]

    for q in raw_queries or []:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(q)

    return cleaned[:MAX_SUB_QUERIES]


def query_decomposer_agent(state) -> dict:
    """
    Expand state['merged'] into several corpus-worded retrieval phrases, one
    per distinct OSHA duty the described work implies. This is a vocabulary-
    translation task, not a reasoning task, and it sits on the retrieval
    critical path -- kept to a single cheap constrained call.

    Never returns an empty list: any failure degrades to today's
    single-query behaviour (search with `merged` alone).
    """
    merged = state.get("merged") or ""

    if not merged.strip():
        return {"sub_queries": [merged]}

    messages = [
        SystemMessage(content=query_decomposer_system_prompt),
        HumanMessage(content=query_decomposer_human_prompt(merged)),
    ]

    try:
        result = decomposer_llm.constrained_invoke(
            messages,
            fallback_order=FALLBACK_ORDER,
            constraine_model=QueryDecomposition,
            method="json_schema",
            groq_reasoning_effort="low",
        )

        return {"sub_queries": _normalize_sub_queries(result.get("sub_queries"), merged)}

    except Exception as e:
        return {
            "sub_queries": [merged],
            "decomposer_error": str(e),
        }
