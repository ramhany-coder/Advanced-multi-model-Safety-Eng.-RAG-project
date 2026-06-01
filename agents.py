import base64
import io
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeRerank
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
from presidio_image_redactor import ImageRedactorEngine
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
image_pii = ImageRedactorEngine()

def hyb_retriver_agent (state:State) -> str :
    query = state.get('merged')
    k = state.get('k',5)

    dense_ret = vbd_ret.as_retriever(kwargs=10)
    sparse_ret  = PersistentBM25Retriever().from_persist_dir(save_dir='osha_sparse',k=5)

    hybrid_ret = EnsembleRetriever(retrievers=[dense_ret,sparse_ret],weights=[0.6,0.4])
    respond = hybrid_ret.invoke(query)

    reranker = PineconeRerank(model='bge-reranker-v2-m3',top_n=k)
    reranked_response = reranker.compress_documents(documents=respond,query=query)

    return{'context':reranked_response}

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
    return{"eng_query":respond.text}

def check_cache_agent(state:State) -> dict[str,any] :
    query = state.get('merged')

    result = cache.get(query)
    if result :
        return {'cached':True,
                "response":result}
    else :
        return {"cached":False}
    
def query_pii_agent(state:State) -> str :
    query = state.get('query')
    lang_code = state.get('language_code',"en")

    ana_result = analyzer.analyze(
        text=query,
        language=lang_code
    )
    anon_results = anonymizer.anonymize(query,ana_result)
    cl_text = anon_results.text
    return {"clean_query" : cl_text}

def image_pii_agent (state:State) ->str:
    image = state.get('image_bytes')

    image_data = base64.b64decode(image)
    pil_image = Image.open(io.BytesIO(image_data))

    red_result = image_pii.redact(image=pil_image,fill="black")

    buffered = io.BytesIO()
    red_result.save(buffered, format="JPEG")
    clean_img_bytes_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return{"image_bytes_cleaned" : clean_img_bytes_base64}

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

def web_scrapper_agent (state:State) -> str :
    query = state.get('merged')
    k = state.get('k',3)

    tool = TavilySearch(kwargs=k)
    respond = tool.invoke(query)

    return {'context':respond}

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

def response_translator (state:State):
    response = state.get('response')
    language = state.get('language')
    lang_code = state.get('language_code')

    messages = [
        SystemMessage(content=response_translator_system_prompt),
        HumanMessage(content=response_translator_human_prompt(english_response=response,target_language=language,target_language_code=lang_code))
    ]
    response = llm.invoke(messages)
    return {"native_response":response.text}

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
