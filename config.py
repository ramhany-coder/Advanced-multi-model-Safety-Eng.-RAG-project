from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Use PascalCase for the class name (Settings) to avoid naming conflicts with the instance
class Settings(BaseSettings):
    GPT_API: str
    GEMINI_API: str
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    # Fixed field names to match your variables
    GROQ_API: str   
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    PARENT_PATH: str

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_PATH: str = "models/embedding"
    EMBEDDINGS_CACHE_DIR: str = "cache/osha_chroma"

    # When True, a cache hit is not returned blindly: an LLM checks that the
    # cached response still fits the live query before it is reused, since the
    # cache lookup can match on a near-duplicate rather than identical past
    # query. Set to False to restore the old behavior (always trust a hit).
    ENABLE_CACHE_REASONING: bool = True

    WHISPER_MODEL_SIZE: str = "small"
    WHISPER_MODEL_PATH: str = "models/whisper"

    PII_TRANSFORMER_MODEL_NAME: str = "Davlan/xlm-roberta-base-ner-hrl"
    PII_TRANSFORMER_MODEL_PATH: str = "models/pii_transformer"
    # 2. Use SettingsConfigDict instead of ConfigDict for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

# 3. Instantiate outside of the class
settings = Settings()