from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.Rewrite.prompts import rewrite_human_prompt, rewrite_system_prompt

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-20b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

rewrite_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def rewrite_agent(state) -> dict:
    query = state.get("eng_query") or ""
    chat_hist = state.get("chat_hist") or []

    messages = [
        SystemMessage(content=rewrite_system_prompt),
        HumanMessage(
            content=rewrite_human_prompt(
                english_normalized_payload=query,
                chat_hist=chat_hist
            )
        )
    ]

    response = rewrite_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {
        "rewritten_query": response
    }
