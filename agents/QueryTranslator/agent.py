from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.QueryTranslator.prompts import query_translator_human_prompt, query_translator_system_prompt

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "llama-3.1-8b-instant"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

query_translator_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def user_query_translator(state) -> dict:
    query = state.get("clean_query") or ""
    audio_transcript = (
        state.get("clean_audio_transcript")
        or state.get("audio_transcript")
        or ""
    )
    user_lang = state.get("language") or "Unknown"
    audio_lang = state.get("detected_voice_language")

    messages = [
        SystemMessage(content=query_translator_system_prompt),
        HumanMessage(
            content=query_translator_human_prompt(
                clean_query=query,
                audio_transcript=audio_transcript,
                detected_query_language=user_lang,
                detected_voice_language=audio_lang
            )
        )
    ]

    response = query_translator_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {
        "eng_query": response
    }
