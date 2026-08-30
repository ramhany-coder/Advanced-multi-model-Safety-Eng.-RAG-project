"""Registry of runtime-editable agent system prompts.

Used by the Streamlit "Prompt Editor" tab so prompts can be tweaked and
re-tested live, without restarting the app.

Each agent module does `from agents.X.prompts import some_prompt`, which binds
a name inside that agent module's own globals. Functions look up globals at
call time, so to actually change what an agent sends to the LLM we must patch
the attribute on the module that owns the live reference read at call time:
- Plain string prompts imported directly into an agent.py: patch the agent
  module (that's the name the agent function actually reads).
- Prompts built from a module-level template string by a function defined in
  prompts.py (DocIdMapper, Reranker): patch the prompts module, since that
  function reads the template from its own module's globals.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field


@dataclass
class PromptField:
    key: str
    agent: str
    label: str
    module_path: str
    attr: str
    help: str = ""
    required_placeholders: tuple[str, ...] = ()

    def _module(self):
        return importlib.import_module(self.module_path)

    def get(self) -> str:
        return getattr(self._module(), self.attr)

    def set(self, value: str) -> None:
        setattr(self._module(), self.attr, value)

    def missing_placeholders(self, value: str) -> list[str]:
        return [p for p in self.required_placeholders if p not in value]


PROMPT_FIELDS: list[PromptField] = [
    PromptField(
        "merger_system", "Merger", "System prompt",
        "agents.Merger.agent", "system_merging_prompt",
    ),
    PromptField(
        "rewrite_system", "Rewrite", "System prompt",
        "agents.Rewrite.agent", "rewrite_system_prompt",
    ),
    PromptField(
        "query_translator_system", "QueryTranslator", "System prompt",
        "agents.QueryTranslator.agent", "query_translator_system_prompt",
    ),
    PromptField(
        "image_analysis_system", "ImageAnalysis", "System prompt",
        "agents.ImageAnalysis.agent", "image_system_prompt",
    ),
    PromptField(
        "responser_system", "Responser", "System prompt",
        "agents.Responser.agent", "responser_system_prompt",
    ),
    PromptField(
        "ranker_system", "Ranker", "System prompt",
        "agents.Ranker.agent", "ranker_system_prompt",
    ),
    PromptField(
        "response_translator_system", "ResponseTranslator", "System prompt",
        "agents.ResponseTranslator.agent", "response_translator_system_prompt",
    ),
    PromptField(
        "doc_id_mapper_system_template", "DocIdMapper", "System prompt template",
        "agents.DocIdMapper.prompts", "doc_id_mapping_system_prompt_template",
        help="Formatted with .format(examples_block=...) before use -- keep the "
             "{examples_block} placeholder somewhere in the text.",
        required_placeholders=("{examples_block}",),
    ),
    PromptField(
        "reranker_system_template", "Reranker", "System prompt template",
        "agents.Reranker.prompts", "reranker_system_prompt_template",
        help="Formatted with .format(top_k=...) before use -- keep the "
             "{top_k} placeholder somewhere in the text.",
        required_placeholders=("{top_k}",),
    ),
]

_DEFAULTS: dict[str, str] = {}


def get_defaults() -> dict[str, str]:
    """Snapshot of each prompt's original text.

    Captured lazily on first call (before any edit could have happened) and
    cached at module level, since Streamlit re-executes the whole script on
    every interaction -- a plain module-level dict is what makes the snapshot
    survive across reruns within the same server process.
    """
    for pf in PROMPT_FIELDS:
        _DEFAULTS.setdefault(pf.key, pf.get())
    return _DEFAULTS
