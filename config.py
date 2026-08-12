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
    CHILD_PATH: str
    CHILD_DOCUMENTS_PATH: str = "osha_extractive_summaries.json"

    PINECONE_MODEL : str
    EMBEDDING_MODEL_NAME : str
    EMBEDDING_MODEL_PATH : str
    # 2. Use SettingsConfigDict instead of ConfigDict for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

# 3. Instantiate outside of the class
settings = Settings()