import json


def _load_registry_by_section_id(registry_path: str) -> dict:
    """Index the parent-doc registry by section_id (e.g. '1926.451')."""
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if isinstance(registry, dict):
        entries = registry.items()
    elif isinstance(registry, list):
        entries = enumerate(registry)
    else:
        raise TypeError("registry.json must be either dict or list.")

    return {
        (item.get("section_id") or str(key)): item
        for key, item in entries
    }


def sample_section_id_examples(registry_path: str, n: int = 10) -> str:
    """
    Build a few-shot block of real section_id/title pairs from the DB,
    evenly spread across the corpus so the LLM sees the ID format plus a
    representative spread of topics rather than just the first few sections.
    """
    by_section_id = _load_registry_by_section_id(registry_path)
    section_ids = sorted(by_section_id.keys())

    if not section_ids:
        return "(no example sections available)"

    step = max(1, len(section_ids) // n)
    sampled = section_ids[::step][:n]

    return "\n".join(
        f"- {sid}: {by_section_id[sid].get('title', '')}" for sid in sampled
    )


def known_section_ids(registry_path: str) -> set[str]:
    """The full set of section_ids that actually exist in the DB, used to drop hallucinated IDs."""
    return set(_load_registry_by_section_id(registry_path).keys())
