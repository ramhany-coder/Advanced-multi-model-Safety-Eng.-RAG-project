from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from pathlib import Path
from loggers import setup_logger
from interfaces.Embedding import EmbeddingInterface
from langchain_core.embeddings import Embeddings
from model_manager import ensure_model_downloaded, SENTENCE_TRANSFORMER_IGNORE_PATTERNS

logger = setup_logger(__name__)
MODEL_PATH = Path(settings.EMBEDDING_MODEL_PATH)

class Embedding_Model(EmbeddingInterface):

    embedding_function = Embeddings
    def __init__(self, model_path: str = MODEL_PATH):
        # Downloads to model_path only the first time; a later restart finds
        # the weights already on disk and loads fully offline.
        model_path = ensure_model_downloaded(
            settings.EMBEDDING_MODEL_NAME,
            model_path,
            ignore_patterns=SENTENCE_TRANSFORMER_IGNORE_PATTERNS,
        )
        logger.info("Loading embedding model from %s", model_path)

        self.embedding_function = HuggingFaceEmbeddings(
                    model_name=model_path,  # local directory
                    model_kwargs={
                        "device": "cpu",
                        "local_files_only": True,
                    })
        return

    def get_client(self):
        return self.embedding_function

local_embedding = Embedding_Model()


