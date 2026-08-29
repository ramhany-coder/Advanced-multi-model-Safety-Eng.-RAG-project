"""Offline build step for the OSHA parent-document embedding cache.

Embeds every section in parent_store/registry.json with the configured
HuggingFace embedding model and persists a Chroma store to
settings.EMBEDDINGS_CACHE_DIR (agents/Retrieve/helpers.py loads it at
runtime and never builds it itself).

Run this locally whenever parent_store/registry.json or the embedding model
changes, then commit the resulting cache directory so it ships with the repo
and the container pulls it pre-built at deploy time instead of embedding the
corpus in production:

    python scripts/build_osha_embeddings.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.Retrieve.embedding_cache import build_embedding_cache
from config import settings

if __name__ == "__main__":
    build_embedding_cache()
    print(f"OSHA embedding cache built at: {settings.EMBEDDINGS_CACHE_DIR}")
