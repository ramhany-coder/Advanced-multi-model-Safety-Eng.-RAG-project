from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.helpers import combine_evidence
from agents.Ranker.prompts import ranker_humman_prompt, ranker_system_prompt
from agents.Ranker.schemas import RankScore

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "llama-3.1-8b-instant"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

ranker_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def ranker_agent(state) -> dict:
    query = state.get("eng_query")
    image = state.get("image_exp")
    response = state.get("response")
    content = combine_evidence(state)

    messages = [
        SystemMessage(content=ranker_system_prompt),
        HumanMessage(content=ranker_humman_prompt(query, image, response, content))
    ]

    try:
        result = ranker_llm.constrained_invoke(
            messages, fallback_order=FALLBACK_ORDER, constraine_model=RankScore
        )

        return {
            "rank": int(result["k"])
        }

    except Exception as e:
        return {
            "rank": 0,
            "ranker_error": str(e)
        }


def rejection_response_agent(state) -> dict:
    """
    Safe fallback response when the QA ranker rejects the generated answer.
    """
    rank_value = state.get("rank", "unknown")

    fallback = (
        "I could not generate a sufficiently reliable OSHA-based compliance answer "
        "from the retrieved context. The QA ranker marked the answer as low confidence "
        f"(rank: {rank_value}). Please provide a clearer image, more site details, "
        "or a more specific safety question so the system can retrieve stronger evidence."
    )

    return {
        "response": fallback,
        "rejected": True
    }
