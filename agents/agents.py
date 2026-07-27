import os
import io
import json
import base64
import tempfile
import sys
from pathlib import Path
from PIL import Image 
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever

# ----------------------------------------------------------------------
# 1. NEW FALLBACK TECHNOLOGY ASSIGNMENTS
# Assign your primary and secondary models here. 
# Available routers (based on your FallBack class): "groq", "gpt", "gemini", "ollama"
# ----------------------------------------------------------------------
PRIMARY_ROUTER = "groq"
PRIMARY_MODEL = "llama-3.1-8b-instant"

SECONDARY_ROUTER = "gpt"
SECONDARY_MODEL = "gpt-4o-mini"

FALLBACK_ORDER = [PRIMARY_ROUTER, SECONDARY_ROUTER]

# Import your custom fallback class
# (Adjust this import based on the actual filename of your fallback module)
from fallback import FallBack  
# ----------------------------------------------------------------------

try:
    from langchain_pinecone import PineconeRerank
except Exception:
    PineconeRerank = None

try:
    import langchain_community.retrievers as community_retrievers
    sys.modules["langchain.retrievers"] = community_retrievers
except Exception:
    pass

try:
    from bm25_retriever.retriever import PersistentBM25Retriever
except Exception:
    PersistentBM25Retriever = None

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

try:
    from presidio_image_redactor import ImageRedactorEngine
except Exception:
    ImageRedactorEngine = None

from gptcache import cache
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt

try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None
    audio_transcription_error = str(e)

from lingua import Language, LanguageDetectorBuilder
from prompt import *
from models import *  # This imports descion, rank, etc.

load_dotenv()

# ----------------------------------------------------------------------
# 2. INITIALIZE FALLBACK MANAGERS
# ----------------------------------------------------------------------
# We pass the dynamically assigned routers/models into kwargs
fallback_kwargs = {
    f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
    f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL
}

# Manager for regular text tasks
text_llm = FallBack(**fallback_kwargs)

# Manager for constrained decision output
decision_llm = FallBack(**fallback_kwargs, constraine_model=descion)

# Manager for constrained ranking output
rank_llm = FallBack(**fallback_kwargs, constraine_model=rank)

# ----------------------------------------------------------------------

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cache.init(pre_embedding_func=get_prompt)
vbd_ret = Chroma(embedding_function=emb, persist_directory='osha')

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
image_pii = ImageRedactorEngine() if ImageRedactorEngine is not None else None

from whisper import stt_model


def audio_transcription_agent(state) -> dict:
    audio_bytes = state.get('audio_bytes', "")
    audio_format = state.get('audio_format', "")
    
    transcript_final, info = stt_model.transcript(
        audio_bytes=audio_bytes,
        audio_format=audio_format
    )
    return {
        "raw_audio_transcript": transcript_final.strip(),
        "detected_voice_language": info.language
    }

def safe_rerank(documents, query: str, k: int):
    """
    Rerank documents if PineconeRerank is available.
    Otherwise return the first k documents.
    """
    if not documents:
        return []

    if PineconeRerank is None:
        return documents[:k]

    try:
        reranker = PineconeRerank(
            model="bge-reranker-v2-m3",
            top_n=k
        )
        return reranker.compress_documents(
            documents=documents,
            query=query
        )
    except Exception:
        return documents[:k]
        
def load_parent_docstore(registry_path: str = "parent_store/registry.json"):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    parent_docstore = InMemoryStore()
    items = []

    if isinstance(registry, dict):
        for doc_id, item in registry.items():
            page_content = item.get("page_content") or item.get("full_text") or ""
            metadata = item.get("metadata") or {}
            metadata["doc_id"] = str(doc_id)
            items.append((
                str(doc_id),
                Document(page_content=page_content, metadata=metadata)
            ))

    elif isinstance(registry, list):
        for i, item in enumerate(registry):
            doc_id = str(item.get("doc_id") or item.get("parent_index") or i)
            page_content = item.get("page_content") or item.get("full_text") or ""
            metadata = dict(item)
            metadata["doc_id"] = doc_id
            items.append((
                doc_id,
                Document(page_content=page_content, metadata=metadata)
            ))

    else:
        raise TypeError("registry.json must be either dict or list.")

    parent_docstore.mset(items)
    return parent_docstore

def hyb_retriver_agent(state) -> dict:
    query = state.get("merged") or ""
    k = int(state.get("k", 5) or 5)

    # Protect query length (assuming clamp_text is imported/defined elsewhere)
    query = clamp_text(query)

    vbd_ret = Chroma(
        collection_name="production_parent_child_store",
        embedding_function=emb,
        persist_directory="osha"
    )

    parent_docstore = load_parent_docstore("parent_store/registry.json")

    dense_ret = vbd_ret.as_retriever(
        search_kwargs={"k": max(10, k*3)}
    )

    def children_to_parents(child_docs, max_parents):
        parent_docs = []
        seen_doc_ids = set()
        for child in child_docs:
            doc_id = child.metadata.get("doc_id")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            parent = parent_docstore.mget([doc_id])[0]
            if parent:
                parent.metadata = dict(parent.metadata or {})
                parent.metadata["matched_child_chunk_id"] = child.metadata.get("chunk_id")
                parent.metadata["matched_child_chunk_type"] = child.metadata.get("chunk_type")
                parent_docs.append(parent)
            if len(parent_docs) >= max_parents:
                break
        return parent_docs

    if PersistentBM25Retriever is None:
        child_docs = dense_ret.invoke(query)
        parent_docs = children_to_parents(child_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_parent"
        }

    try:
        sparse_ret = PersistentBM25Retriever.load(save_dir="osha_sparse")
        sparse_ret.k = max(5, k)

        hybrid_ret = EnsembleRetriever(
            retrievers=[dense_ret, sparse_ret],
            weights=[0.6, 0.4]
        )
        retrieved_docs = hybrid_ret.invoke(query)
        parent_docs = children_to_parents(retrieved_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "hybrid_child_to_parent"
        }

    except Exception as e:
        child_docs = dense_ret.invoke(query)
        parent_docs = children_to_parents(child_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_parent_after_bm25_error",
            "bm25_error": str(e)
        }

language_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.ARABIC,
    Language.FRENCH,
    Language.SPANISH,
    Language.GERMAN
).build()


def local_language_detector_agent(state) -> dict:
    query = state.get("query", "")
    detected = language_detector.detect_language_of(query)

    if detected is None:
        return {
            "language": "English",
            "language_code": "en",
            "origin_en": True
        }

    lang_name = detected.name.capitalize()
    lang_code = detected.iso_code_639_1.name.lower()

    return {
        "language": lang_name,
        "language_code": lang_code,
        "origin_en": lang_code == "en"
    }
    
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

    # Use the FallBack text manager
    response = text_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {
        "eng_query": response
    }

def check_cache_agent(state) -> dict[str, any]:
    query = state.get('merged')
    result = cache_get(query)
    if result:
        return {'cached': True, "response": result}
    else:
        return {"cached": False}

def redact_text_with_presidio(text: str) -> str:
    if not text:
        return ""
    if analyzer is None:
        return text
    try:
        results = analyzer.analyze(text=text, language="en")
        anon = anonymizer.anonymize(text=text, analyzer_results=results)
        return anon.text
    except Exception:
        return text
        
def query_pii_agent(state) -> dict:
    query = state.get("query") or ""
    audio_transcript = state.get("audio_transcript") or ""

    clean_query = redact_text_with_presidio(query)
    clean_audio_transcript = redact_text_with_presidio(audio_transcript)

    return {
        "clean_query": clean_query,
        "clean_audio_transcript": clean_audio_transcript,
        "pii_language_used": "en"
    }
        
def image_pii_agent(state) -> dict:
    image = state.get("image_bytes")

    if not image:
        return {"image_bytes_cleaned": None}

    if image_pii is None:
        return {
            "image_bytes_cleaned": image,
            "image_redaction_mode": "passthrough_no_redactor"
        }

    try:
        image_data = base64.b64decode(image)
        pil_image = Image.open(io.BytesIO(image_data))

        red_result = image_pii.redact(image=pil_image, fill="black")
        buffered = io.BytesIO()
        red_result.save(buffered, format="JPEG")
        clean_img_bytes_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image_bytes_cleaned": clean_img_bytes_base64,
            "image_redaction_mode": "presidio_redacted"
        }
    except Exception:
        return {
            "image_bytes_cleaned": image,
            "image_redaction_mode": "passthrough_after_error"
        }

def rewrite_agent(state) -> dict:
    query = state.get("eng_query") or ""
    chat_hist = state.get("chat_hist") or []

    messages = [
        SystemMessage(content=rewrite_system_prompt),
        HumanMessage(
            content=rewrite_human_prompt(
                english_normalized_payload=query,
                chat_hist=chat_hist
            )
        )
    ]

    # Use the FallBack text manager
    response = text_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {
        "rewritten_query": response
    }

def image_exp_agent(state) -> str:
    img = state.get('image_bytes_cleaned')

    messages = [
        SystemMessage(content=image_system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this asset for compliance evaluation."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        ])
    ]

    # Use the FallBack text manager
    response = text_llm.invoke(messages, fallback_order=FALLBACK_ORDER)
    return {'image_exp': response}

def merging_agent(state) -> str:
    query = state.get('rewritten_query')
    img = state.get('image_exp')

    if img:
        messages = [
            SystemMessage(content=system_merging_prompt),
            HumanMessage(content=merging_humman_prompt(query, img))
        ]

        # Use the FallBack text manager
        response = text_llm.invoke(messages, fallback_order=FALLBACK_ORDER)
        return {'merged': response}
    else:
        return {'merged': query}

def k_getter_use_web(state) -> dict:
    query = state.get("merged") or ""
    forced_is_web = state.get("is_web", None)
    has_forced_is_web = forced_is_web is not None

    messages = [
        SystemMessage(content=k_web_system_prompt),
        HumanMessage(content=k_web_humman(query)) 
        # Note: I removed the extra JSON parsing suffix because .with_structured_output() handles it natively
    ]

    try:
        # Use the FallBack Constrained Invoke Manager 
        # (It will return a dictionary directly based on your Pydantic descion schema)
        results = decision_llm.constrained_invoke(messages, fallback_order=FALLBACK_ORDER)

        return {
            "k": int(results["k"]),
            "is_web": bool(forced_is_web) if has_forced_is_web else bool(results["is_web"])
        }

    except Exception as e:
        return {
            "k": 8,
            "is_web": bool(forced_is_web) if has_forced_is_web else False,
            "structured_output_error": str(e)
        }

def web_scrapper_agent(state) -> dict:
    query = state.get("merged") or ""
    k = state.get("k", 8)

    try:
        tool = TavilySearch(max_results=k)
        respond = tool.invoke(query)
        return {
            "context": respond,
            "retrieval_mode": "web_search"
        }
    except Exception:
        return {
            "context": [],
            "retrieval_mode": "web_search_failed"
        }

def responser_agent(state) -> str:
    query = state.get('merged')
    context = state.get('context', [])
    
    messages = [
        SystemMessage(content=responser_system_prompt),
        HumanMessage(content=responser_humman_prompt(query, context))
    ]

    # Use the FallBack text manager
    response = text_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {'response': response}

def response_translator(state) -> dict:
    response = state.get("response") or ""
    language = state.get("language") or "English"
    lang_code = state.get("language_code") or "en"

    if lang_code == "en":
        return {
            "native_response": response,
            "final_response": response
        }

    messages = [
        SystemMessage(content=response_translator_system_prompt),
        HumanMessage(
            content=response_translator_human_prompt(
                english_response=response,
                target_language=language,
                target_language_code=lang_code
            )
        )
    ]

    # Use the FallBack text manager
    translated = text_llm.invoke(messages, fallback_order=FALLBACK_ORDER)

    return {
        "native_response": translated
    }

def caching_agent(state) -> dict[str, any]:
    caching_stat = state.get('cached')
    if not caching_stat:
        query = state.get('merged')
        response = state.get('response')
        if response and query:
            cache_put(query, response)


def ranker_agent(state) -> dict:
    query = state.get("eng_query")
    image = state.get("image_exp")
    response = state.get("response")
    content = state.get("context")

    messages = [
        SystemMessage(content=ranker_system_prompt),
        HumanMessage(content=ranker_humman_prompt(query, image, response, content))
        # Note: Extra JSON manual parsing prompt was removed, handled by schema constraints
    ]

    try:
        # Use the FallBack Constrained Invoke Manager bound to 'rank'
        result = rank_llm.constrained_invoke(messages, fallback_order=FALLBACK_ORDER)

        return {
            "rank": int(result["k"])  # Assuming 'k' holds the rank score as per original code logic
        }

    except Exception as e:
        return {
            "rank": 0,
            "ranker_error": str(e)
        }

def rejection_response_agent(state) -> dict:
    """
    Safe fallback response when the QA ranker rejects the generated answer.
    """
    rank_value = state.get("rank", "unknown")

    fallback = (
        "I could not generate a sufficiently reliable OSHA-based compliance answer "
        "from the retrieved context. The QA ranker marked the answer as low confidence "
        f"(rank: {rank_value}). Please provide a clearer image, more site details, "
        "or a more specific safety question so the system can retrieve stronger evidence."
    )

    return {
        "response": fallback,
        "rejected": True
    }