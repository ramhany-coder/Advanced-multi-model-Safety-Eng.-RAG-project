import os

from langchain_core.documents import Document
from config import settings

try:
    from langchain_pinecone import PineconeRerank
except Exception:
    PineconeRerank = None


class Pinecone:
    def __init__(self, model: str, k: int):
        if PineconeRerank is None:
            raise ValueError("langchain_pinecone is not installed; Pinecone reranker unavailable.")

        try:
            self.reranker = PineconeRerank(
                model=model,
                top_k=k
            )
        except Exception as e:
            raise ValueError(f"Error in loading Pinecone reranker model :{str(e)}")

    def rerank(self, documents, query: str):
        try:
            return self.reranker.compress_documents(
                        documents=documents,
                        query=query)
        except Exception as e:
            raise ValueError(f"Error during the reranking step by Pinecone : {str(e)}")


def pinecone_is_configured() -> bool:
    """Router switch: Pinecone rerank is only usable when its API key is set."""
    return bool(os.getenv("PINECONE_API_KEY")) and PineconeRerank is not None


def _build_pinecone_client(k: int = 8):
    if not pinecone_is_configured():
        return None
    try:
        return Pinecone(settings.PINECONE_MODEL, k)
    except Exception:
        return None


# None when PINECONE_API_KEY is not set (or the package/model failed to load),
# so importing this module never crashes when Pinecone isn't configured.
pinecone_client = _build_pinecone_client()