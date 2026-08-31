from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from agents.helpers import validate_router
from config import settings

class Llm :
    routers_list = ["gemini","gpt","groq","ollama"]
    
    def __init__(self,temp:float=0):
        self.temp = temp

    def get_model(self, router: str, model: str, **kwargs):
        router = validate_router(router)

        providers = {
            "gemini": Llm.gemini,
            "groq": Llm.groq,
            "ollama": Llm.ollama,
            "gpt": Llm.gpt,
        }

        return providers[router](model, self.temp, **kwargs)
        
# 1. Google Gemini
    @staticmethod
    def gemini(model: str, temp: float):
        return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GEMINI_API,
        temperature=temp,
        )


    # 2. Groq
    @staticmethod
    def groq(model: str = "qwen/qwen3.6-27b", temp: float = None, reasoning_effort: str | None = None):
        kwargs = {}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        return ChatGroq(
        model=model,
        api_key=settings.GROQ_API,
        temperature=temp,
        # gpt-oss models spend part of this budget on a hidden reasoning
        # channel before writing the final answer; left unset, Groq's
        # server-side default is too small and the response gets cut off
        # mid-reasoning, leaving nothing valid to parse or tool-call with.
        # Kept well under this account's ~8000 TPM cap (shared across every
        # agent in the pipeline) so one call doesn't starve the others.
        max_tokens=4096,
        **kwargs,
        )


    # 3. Local Ollama
    @staticmethod
    def ollama(model: str, temp: float):
        return ChatOllama(
        model=model,
        base_url=settings.OLLAMA_PATH,
        temperature=temp,
        )


    # 4. OpenAI GPT
    @staticmethod
    def gpt(model: str, temp: float):
        return ChatOpenAI(
        model=model,
        api_key=settings.GPT_API,
        temperature=temp,
        )
    
client_llm = Llm(temp=0.1)