import os
import io
import base64
import tempfile
from openai import OpenAI
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

def audio_transcription_agent(state: State) -> dict:
    """
    Transcribes uploaded audio into text.

    Expected state input:
        audio_bytes: base64-encoded audio file
        audio_format: mp3 / wav / m4a / webm / ogg

    Output:
        audio_transcript: clean transcript text

    Notes:
    - This node does not overwrite the typed user query.
    - The query translator should later combine:
        clean_query + clean_audio_transcript / audio_transcript
    """

    audio_b64 = state.get("audio_bytes")
    audio_format = (state.get("audio_format") or "mp3").lower().replace(".", "")

    if not audio_b64:
        return {
            "audio_transcript": ""
        }

    supported_formats = {"mp3", "wav", "m4a", "webm", "ogg", "mpeg", "mpga"}

    if audio_format not in supported_formats:
        audio_format = "mp3"

    temp_path = None

    try:
        audio_data = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False
        ) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()
            temp_path = temp_audio.name

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=os.environ.get(
                    "OPENAI_TRANSCRIBE_MODEL",
                    "gpt-4o-mini-transcribe"
                ),
                file=audio_file,
                prompt=(
                    "Transcribe this construction safety field note accurately. "
                    "Preserve OSHA references, equipment names, measurements, "
                    "Arabic/English mixed speech meaning, and uncertainty. "
                    "Do not answer the question."
                )
            )

        raw_transcript = getattr(transcription, "text", str(transcription)).strip()

        # Optional cleanup using your existing LLM and prompt.py prompts.
        # If cleanup fails, return raw transcript.
        try:
            cleanup_messages = [
                SystemMessage(content=audio_transcription_system_prompt),
                HumanMessage(
                    content=(
                        f"{audio_transcription_human_prompt()}\n\n"
                        f"Raw Transcript:\n{raw_transcript}\n\n"
                        "Clean Transcript:"
                    )
                )
            ]

            cleaned = llm.invoke(cleanup_messages).content.strip()

            return {
                "audio_transcript": cleaned,
                "raw_audio_transcript": raw_transcript,
                "audio_transcription_status": "success_cleaned"
            }

        except Exception:
            return {
                "audio_transcript": raw_transcript,
                "raw_audio_transcript": raw_transcript,
                "audio_transcription_status": "success_raw"
            }

    except Exception as e:
        return {
            "audio_transcript": "",
            "audio_transcription_status": "failed",
            "audio_transcription_error": str(e)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


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
    
def user_query_translator(state: State) -> dict:
    query = state.get("clean_query") or ""
    audio_transcript = (
        state.get("clean_audio_transcript")
        or state.get("audio_transcript")
        or ""
    )
    lang = state.get("language") or "Unknown"

    messages = [
        SystemMessage(content=query_translator_system_prompt),
        HumanMessage(
            content=query_translator_human_prompt(
                clean_query=query,
                audio_transcript=audio_transcript,
                detected_language=lang
            )
        )
    ]

    respond = llm.invoke(messages)

    return {
        "eng_query": respond.content
    }

def check_cache_agent(state:State) -> dict[str,any] :
    query = state.get('merged')

    result = cache.get(query)
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
    query = state.get('eng_query')
    image = state.get('image_exp')
    response = state.get('response')
    content = state.get('context')

    messages = [
        SystemMessage(content=ranker_system_prompt),
        HumanMessage(content=ranker_humman_prompt(query,image,response,content))
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
