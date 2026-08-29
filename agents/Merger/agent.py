from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.Merger.prompts import merging_humman_prompt, system_merging_prompt

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "llama-3.1-8b-instant"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

merger_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def merging_agent(state) -> dict:
    query = state.get('rewritten_query')
    img = state.get('image_exp')

    if not img:
        return {'merged': query}

    messages = [
        SystemMessage(content=system_merging_prompt),
        HumanMessage(content=merging_humman_prompt(query, img))
    ]

    response = merger_llm.invoke(messages, fallback_order=FALLBACK_ORDER)
    return {'merged': response}
