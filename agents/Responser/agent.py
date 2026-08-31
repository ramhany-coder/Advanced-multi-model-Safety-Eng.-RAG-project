import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.helpers import combine_evidence, format_context_for_prompt
from agents.Responser.prompts import responser_humman_prompt, responser_system_prompt

logger = logging.getLogger("pipeline")

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-safeguard-20b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

# Groq's free on_demand tier caps this org at 8000 tokens/minute, so a full,
# unclamped context block can get the request rejected outright before it
# even runs. Only the groq attempt gets capped -- gpt has more headroom and
# should still see everything.
GROQ_CONTEXT_MAX_DOCS = 10
GROQ_CONTEXT_MAX_CHARS_PER_DOC = 900

responser_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def responser_agent(state) -> dict:
    query = state.get('merged')
    context = combine_evidence(state)

    def build_messages(router: str):
        if router == "groq":
            context_text = format_context_for_prompt(
                context,
                max_docs=GROQ_CONTEXT_MAX_DOCS,
                max_chars_per_doc=GROQ_CONTEXT_MAX_CHARS_PER_DOC,
            )
        else:
            context_text = format_context_for_prompt(context)
        return [
            SystemMessage(content=responser_system_prompt),
            HumanMessage(content=responser_humman_prompt(query, context_text))
        ]

    response = responser_llm.invoke(build_messages, fallback_order=FALLBACK_ORDER)

    logger.info(
        "RESPONSER RAW OUTPUT (%d chars, %d context docs):\n%s",
        len(response or ""), len(context or []), response,
    )
    logger.info(
        "RESPONSER CONTEXT: %s",
        [d.metadata.get("chunk_id") for d in (context or [])],
    )

    return {'response': response}
