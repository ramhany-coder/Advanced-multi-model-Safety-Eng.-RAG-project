# wrap_kv.py
import json
from pathlib import Path

REGISTRY_PATH = Path("parent_store") / "registry.json"

if not REGISTRY_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {REGISTRY_PATH}")

with REGISTRY_PATH.open("r", encoding="utf-8") as f:
    registry = json.load(f)

parents = {}

# Supports both formats:
# 1) {"0": {"page_content": "...", "metadata": {...}}}
# 2) [{"doc_id": "0", "full_text": "...", ...}]
if isinstance(registry, dict):
    for doc_id, item in registry.items():
        page_content = item.get("page_content") or item.get("full_text") or ""
        metadata = item.get("metadata") or {}
        metadata["doc_id"] = str(doc_id)
        parents[str(doc_id)] = {
            "page_content": page_content,
            "metadata": metadata,
        }

elif isinstance(registry, list):
    for item in registry:
        doc_id = str(item.get("doc_id") or item.get("parent_index") or len(parents))
        page_content = item.get("page_content") or item.get("full_text") or ""
        metadata = dict(item)
        metadata["doc_id"] = doc_id
        parents[doc_id] = {
            "page_content": page_content,
            "metadata": metadata,
        }

else:
    raise TypeError("registry.json must be either a dict or a list.")

print(f"✅ Loaded parent registry successfully.")
print(f"✅ Parent documents found: {len(parents)}")

first_id = next(iter(parents))
first_doc = parents[first_id]

print("\nSample parent:")
print("doc_id:", first_id)
print("title:", first_doc["metadata"].get("title"))
print("section_id:", first_doc["metadata"].get("section_id"))
print("text snippet:", first_doc["page_content"][:300].replace("\n", " "), "...")