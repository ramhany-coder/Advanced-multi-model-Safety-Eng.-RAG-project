from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.helpers import combine_evidence
from agents.Responser.prompts import responser_humman_prompt, responser_system_prompt

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "llama-3.1-8b-instant"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

responser_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def responser_agent(state) -> dict:
    query = state.get('merged')
    context = combine_evidence(state)

    messages = [
        SystemMessage(content=responser_system_prompt),
        HumanMessage(content=responser_humman_prompt(query, context))
    ]

    response = responser_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {'response': response}
