from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from pathlib import Path
from loggers import setup_logger
from interfaces.Embedding import EmbeddingInterface
from langchain_core.embeddings import Embeddings

logger = setup_logger(__name__)
MODEL_PATH = Path(settings.EMBEDDING_MODEL_PATH)

class Embedding_Model(EmbeddingInterface):

    embedding_function = Embeddings
    def __init__(self,model_path:str=MODEL_PATH):
        logger.info("Loading embedding model from %s", model_path)

        if not self.validate_path(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.embedding_function = HuggingFaceEmbeddings(
                    model_name=model_path,  # local directory
                    model_kwargs={
                        "device": "cpu",
                        "local_files_only": True,
                    })
        return

    def get_client(self):
        return self.embedding_function

    @staticmethod
    def validate_path(path:str):
        path = Path(path)
        return path

local_embedding = Embedding_Model()


