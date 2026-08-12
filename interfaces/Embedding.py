from abc import abstractmethod
from langchain_core.embeddings import Embeddings
from agents.SingeltonLayer import SingeltonLayer


class EmbeddingInterface (SingeltonLayer):

    @property
    @abstractmethod
    def get_client() -> Embeddings:
        pass