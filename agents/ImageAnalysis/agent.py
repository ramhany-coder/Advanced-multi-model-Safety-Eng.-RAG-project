from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm.fallback import FallBack
from agents.ImageAnalysis.prompts import image_system_prompt

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "qwen/qwen3.8-27b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

image_exp_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def image_exp_agent(state) -> dict:
    img = state.get('image_bytes_cleaned')

    if not img:
        return {
            'image_exp': (
                "Image analysis was skipped because PII redaction could not be "
                "confirmed for the uploaded image (see image_redaction_mode)."
            )
        }

    messages = [
        SystemMessage(content=image_system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this asset for compliance evaluation."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        ])
    ]

    response = image_exp_llm.invoke(messages, fallback_order=FALLBACK_ORDER)
    return {'image_exp': response}
