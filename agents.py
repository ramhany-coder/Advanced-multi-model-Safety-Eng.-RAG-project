import base64
import io
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
except Exception:
    ImageRedactorEngine = None

from gptcache import cache
from lingua import Language, LanguageDetectorBuilder
from prompt import *
from models import *
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model='gpt-4o', temperature=0.2)
emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vbd_ret = Chroma(embedding_function=emb,
                 persist_directory='osha')

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
image_pii = ImageRedactorEngine() if ImageRedactorEngine is not None else None

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
        
def hyb_retriver_agent(state: State) -> dict:
    query = state.get("merged") or ""
    k = state.get("k", 5)

    dense_ret = vbd_ret.as_retriever(
        search_kwargs={"k": 10}
    )

    # If BM25 is not available on Streamlit Cloud, use dense-only fallback
    if PersistentBM25Retriever is None:
        dense_docs = dense_ret.invoke(query)
        reranked_response = safe_rerank(dense_docs, query, k)

        return {
            "context": reranked_response,
            "retrieval_mode": "dense_only_fallback"
        }

    try:
        sparse_ret = PersistentBM25Retriever().from_persist_dir(
            save_dir="osha_sparse",
            k=5
        )

        hybrid_ret = EnsembleRetriever(
            retrievers=[dense_ret, sparse_ret],
            weights=[0.6, 0.4]
        )

        docs = hybrid_ret.invoke(query)
        reranked_response = safe_rerank(docs, query, k)

        return {
            "context": reranked_response,
            "retrieval_mode": "hybrid_dense_sparse"
        }

    except Exception:
        dense_docs = dense_ret.invoke(query)
        reranked_response = safe_rerank(dense_docs, query, k)

        return {
            "context": reranked_response,
            "retrieval_mode": "dense_only_fallback_after_bm25_error"
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
    
def user_query_translator (state:State):
    query = state.get('clean_query')
    lang = state.get('language')

    messages = [
        SystemMessage(content=query_translator_system_prompt),
        HumanMessage(content=query_translator_human_prompt(clean_query=query,detected_language=lang))
    ]

    respond = llm.invoke(messages)
    return{"eng_query":respond.content}

def check_cache_agent(state:State) -> dict[str,any] :
    query = state.get('merged')

    result = cache.get(query)
    if result :
        return {'cached':True,
                "response":result}
    else :
        return {"cached":False}
    
def query_pii_agent(state: State) -> dict:
    query = state.get("query") or ""

    # MVP-safe fallback:
    # Presidio default recognizers are strongest in English.
    # Arabic production support should use custom recognizers.
    presidio_lang = "en"

    try:
        ana_result = analyzer.analyze(
            text=query,
            language=presidio_lang
        )

        anon_results = anonymizer.anonymize(
            text=query,
            analyzer_results=ana_result
        )

        return {
            "clean_query": anon_results.text,
            "pii_language_used": presidio_lang
        }

    except Exception:
        return {
            "clean_query": query,
            "pii_language_used": "fallback_none"
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

def rewrite_agent (state:State) -> str :
    query = state.get('eng_query')
    chat_hist = state.get('chat_hist')

    messages = [
        SystemMessage(content=rewrite_system_prompt),
        HumanMessage(content=rewrite_human_prompt(query,chat_hist))
    ]

    response = llm.invoke(messages)
    res = response.content

    return {'rewritten_query' : res }

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

llm_cons = llm.with_structured_output(descion)
def k_getter_use_web(state:State) -> str:
    query = state.get('merged')

    messages = [ 
        SystemMessage(content=k_web_system_prompt),
        HumanMessage(content=k_web_humman(query))
    ]

    results : descion = llm_cons.invoke(messages)
    return {
        'k' : results.k ,
        'is_web' : results.is_web
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
            cache.import_data([query],[response])

llm_cons_rank = llm.with_structured_output(rank)
def ranker_agent(state:State) -> str :
    query = state.get('clean_query')
    image = state.get('image_bytes_cleaned')
    response = state.get('response')

    messages = [
        SystemMessage(content=ranker_system_prompt),
        HumanMessage(content=ranker_humman_prompt(query,image,response))
    ]

    result : rank = llm_cons_rank.invoke(messages)
    return {'rank': result.k}

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
