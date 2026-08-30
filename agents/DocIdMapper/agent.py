from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from agents.llm.fallback import FallBack
from agents.Retrieve.helpers import load_parent_documents
from agents.DocIdMapper.helpers import known_section_ids, sample_section_id_examples
from agents.DocIdMapper.prompts import doc_id_mapping_human_prompt, doc_id_mapping_system_prompt
from agents.DocIdMapper.schemas import DocIdMapping

PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-20b"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

doc_id_mapper_llm = FallBack(
    **{
        f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
        f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    }
)


def doc_id_mapper_agent(state) -> dict:
    """
    Map the merged retrieval query straight to known OSHA section_ids using the
    LLM's own OSHA knowledge, then fetch those sections directly from the DB.

    Downstream, the workflow only falls back to the full hybrid retriever when
    this agent finds fewer than 3 valid section_ids or flags need_more=True.
    """
    query = state.get("merged") or ""
    registry_path = settings.PARENT_PATH or "parent_store/registry.json"

    messages = [
        SystemMessage(content=doc_id_mapping_system_prompt(sample_section_id_examples(registry_path))),
        HumanMessage(content=doc_id_mapping_human_prompt(query)),
    ]

    try:
        result = doc_id_mapper_llm.constrained_invoke(
            messages, FALLBACK_ORDER, constraine_model=DocIdMapping
        )
        candidate_ids = result.get("section_ids") or []
        need_more = bool(result.get("need_more"))
    except Exception as e:
        return {
            "section_ids": [],
            "need_more": True,
            "content": [],
            "doc_id_mapper_error": str(e),
        }

    valid_ids = known_section_ids(registry_path)
    section_ids = [sid for sid in dict.fromkeys(candidate_ids) if sid in valid_ids]

    matched_docs = (
        load_parent_documents(registry_path, given_section_id=section_ids)
        if section_ids
        else []
    )

    return {
        "section_ids": section_ids,
        "need_more": need_more,
        "content": matched_docs,
    }
