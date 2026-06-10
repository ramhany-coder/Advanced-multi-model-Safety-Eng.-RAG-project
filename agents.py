from pathlib import Path
import io
import os
import base64
import tempfile
from openai import OpenAI
import json
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
try:
    from langchain_pinecone import PineconeRerank
except Exception:
    PineconeRerank = None
    
from langchain_tavily import TavilySearch
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
import sys

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
from PIL import Image 
try:
    from presidio_image_redactor import ImageRedactorEngine
except Exception  :
    ImageRedactorEngine = None

from gptcache import cache
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt

try :
    from faster_whisper import WhisperModel
except Exception as e :
    WhisperModel = None
    audio_transcription_error = str(e)

from lingua import Language, LanguageDetectorBuilder
from prompt import *
from models import *
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
load_dotenv()

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)


emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

cache.init(pre_embedding_func=get_prompt)

vbd_ret = Chroma(embedding_function=emb,
                 persist_directory='osha')

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
image_pii = ImageRedactorEngine() if ImageRedactorEngine is not None else None

def audio_transcription_agent(state: State) -> dict:
    audio_bytes = state.get('audio_bytes',"")
    audio_formate = state.get('audio_format',"")
    
    if WhisperModel is None :
        return {  "raw_audio_transcript" :"",
          "detected_voice_language" :""}
        
    else :
        if audio_bytes :
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=audio_formate
            )
        
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()
            audio_path = temp_file.name
        
            model = WhisperModel("base",device='cpu',compute_type="int8")
        
            segments , info = model.transcribe(audio_path,beam_size=5, vad_filter=True)
        
            transcript_final = " ".join(segment.text.strip() for segment in segments)
            return {
                "raw_audio_transcript" : transcript_final.strip(),
              "detected_voice_language" : info.language
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
    """
    Load parent OSHA documents from registry.json into an InMemoryStore.
    ParentDocumentRetriever will use child metadata doc_id to fetch these parents.
    """
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
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            ))

    elif isinstance(registry, list):
        for i, item in enumerate(registry):
            doc_id = str(item.get("doc_id") or item.get("parent_index") or i)
            page_content = item.get("page_content") or item.get("full_text") or ""
            metadata = dict(item)
            metadata["doc_id"] = doc_id

            items.append((
                doc_id,
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            ))

    else:
        raise TypeError("registry.json must be either dict or list.")

    parent_docstore.mset(items)
    return parent_docstore

def hyb_retriver_agent(state: State) -> dict:
    query = state.get("merged") or ""
    k = int(state.get("k", 5) or 5)

    # Protect query length
    query = clamp_text(query)

    # Load Chroma child chunk store
    vbd_ret = Chroma(
        collection_name="production_parent_child_store",
        embedding_function=emb,
        persist_directory="osha"
    )

    # Load parent document registry
    parent_docstore = load_parent_docstore("parent_store/registry.json")

    # Dense retriever over child chunks
    dense_ret = vbd_ret.as_retriever(
        search_kwargs={"k": max(10, k*3)}
    )

    # Helper: convert child chunks to parent documents
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

    # If BM25 is unavailable, fallback to dense child retrieval
    if PersistentBM25Retriever is None:
        child_docs = dense_ret.invoke(query)
        parent_docs = children_to_parents(child_docs, k)
        reranked_response = safe_rerank(parent_docs, query, k)
        return {
            "context": reranked_response,
            "retrieval_mode": "dense_child_to_parent"
        }

    try:
        # Load sparse BM25 retriever
        sparse_ret = PersistentBM25Retriever.load(save_dir="osha_sparse")
        sparse_ret.k = max(5, k)

        # Hybrid ensemble
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


def local_language_detector_agent(state: State) -> dict:
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
    
def user_query_translator(state: State) -> dict:
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
                audio_transcript=audio_transcript ,
                detected_query_language= user_lang ,
                detected_voice_language = audio_lang ))
    ]

    respond = llm.invoke(messages)

    return {
        "eng_query": respond.content
    }

def check_cache_agent(state:State) -> dict[str,any] :
    query = state.get('merged')

    result = cache_get(query)
    if result :
        return {'cached':True,
                "response":result}
    else :
        return {"cached":False}
def redact_text_with_presidio(text: str) -> str:
    if not text:
        return ""

    if analyzer is None:
        return text

    try:
        results = analyzer.analyze(
            text=text,
            language="en"
        )

        anon = anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        return anon.text

    except Exception:
        return text
        
def query_pii_agent(state: State) -> dict:
    query = state.get("query") or ""
    audio_transcript = state.get("audio_transcript") or ""

    clean_query = redact_text_with_presidio(query)
    clean_audio_transcript = redact_text_with_presidio(audio_transcript)

    return {
        "clean_query": clean_query,
        "clean_audio_transcript": clean_audio_transcript,
        "pii_language_used": "en"
    }
        
def image_pii_agent(state: State) -> dict:
    image = state.get("image_bytes")

    if not image:
        return {
            "image_bytes_cleaned": None
        }

    # If Presidio image redactor / OpenCV fails on Streamlit Cloud,
    # pass the original image through instead of crashing the app.
    if image_pii is None:
        return {
            "image_bytes_cleaned": image,
            "image_redaction_mode": "passthrough_no_redactor"
        }

    try:
        image_data = base64.b64decode(image)
        pil_image = Image.open(io.BytesIO(image_data))

        red_result = image_pii.redact(
            image=pil_image,
            fill="black"
        )

        buffered = io.BytesIO()
        red_result.save(buffered, format="JPEG")

        clean_img_bytes_base64 = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        return {
            "image_bytes_cleaned": clean_img_bytes_base64,
            "image_redaction_mode": "presidio_redacted"
        }

    except Exception:
        return {
            "image_bytes_cleaned": image,
            "image_redaction_mode": "passthrough_after_error"
        }

def rewrite_agent(state: State) -> dict:
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

    response = llm.invoke(messages)

    return {
        "rewritten_query": response.content
    }

def image_exp_agent (state:State) -> str :
    img = state.get('image_bytes_cleaned')

    # Standard LangChain multimodal structure payload
    messages = [
        SystemMessage(content=image_system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this asset for compliance evaluation."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        ])
    ]

    respond = llm.invoke(messages)
    res = respond.content

    return{'image_exp': res}

def merging_agent (state:State) -> str :
    query = state.get('rewritten_query')
    img = state.get('image_exp')

    messages = [
        SystemMessage(content = system_merging_prompt)
        , HumanMessage(content=merging_humman_prompt(query,img))
    ]

    response = llm.invoke(messages)
    res = response.content

    return {'merged' : res }

# llm_cons = llm.with_structured_output(descion)
# def k_getter_use_web(state:State) -> str:
#     query = state.get('merged')

#     messages = [ 
#         SystemMessage(content=k_web_system_prompt),
#         HumanMessage(content=k_web_humman(query))
#     ]

#     results : descion = llm_cons.invoke(messages)
#     return {
#         'k' : results.k ,
#         'is_web' : results.is_web
#     }

llm_json = llm.bind(response_format={"type": "json_object"})
def k_getter_use_web(state: State) -> dict:
    query = state.get("merged") or ""

    messages = [
        SystemMessage(content=k_web_system_prompt),
        HumanMessage(
            content=(
                k_web_humman(query)
                + '\n\nReturn ONLY valid JSON in this exact shape: '
                + '{"k": 3, "is_web": true}'
            )
        )
    ]

    try:
        response = llm_json.invoke(messages)
        data = json.loads(response.content)
        results = descion.model_validate(data)

        return {
            "k": int(results.k),
            "is_web": bool(results.is_web)
        }

    except Exception as e:
        return {
            "k": 3,
            "is_web": False,
            "structured_output_error": str(e)
        }
def web_scrapper_agent(state: State) -> dict:
    query = state.get("merged") or ""
    k = state.get("k", 3)

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

def responser_agent (state:State) -> str:
    query = state.get('merged')
    context = state.get('context', [])
    
    messages = [
        SystemMessage(content=responser_system_prompt),
        HumanMessage(content=responser_humman_prompt(query, context))
    ]

    response = llm.invoke(messages)
    res = response.content

    return {'response': res}

def response_translator(state: State) -> dict:
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

    translated = llm.invoke(messages)

    return {
        "native_response": translated.content
    }
def caching_agent (state:State) -> dict[str,any]:
    caching_stat = state.get('cached')
    if not caching_stat :
        query = state.get('merged')
        response = state.get('response')
        if response and query :
            cache_put(query,response)

# llm_cons_rank = llm.with_structured_output(rank)
# def ranker_agent(state:State) -> str :
#     query = state.get('eng_query')
#     image = state.get('image_exp')
#     response = state.get('response')
#     content = state.get('context')

#     messages = [
#         SystemMessage(content=ranker_system_prompt),
#         HumanMessage(content=ranker_humman_prompt(query,image,response,content))
#     ]

#     result : rank = llm_cons_rank.invoke(messages)
#     return {'rank': result.k}
def ranker_agent(state: State) -> dict:
    query = state.get("eng_query")
    image = state.get("image_exp")
    response = state.get("response")
    content = state.get("context")

    messages = [
        SystemMessage(content=ranker_system_prompt),
        HumanMessage(
            content=(
                ranker_humman_prompt(query, image, response, content)
                + '\n\nReturn ONLY valid JSON in this exact shape: '
                + '{"k": 8}'
            )
        )
    ]

    try:
        result_response = llm_json.invoke(messages)
        data = json.loads(result_response.content)
        result = rank.model_validate(data)

        return {
            "rank": int(result.k)
        }

    except Exception as e:
        return {
            "rank": 0,
            "ranker_error": str(e)
        }
def rejection_response_agent(state: State) -> dict:
    """
    Safe fallback response when the QA ranker rejects the generated answer.
    This response is English only, then response_translator translates it.
    Rejected responses should not be cached.
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
