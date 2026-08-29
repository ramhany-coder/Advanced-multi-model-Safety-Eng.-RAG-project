from abc import abstractmethod
from langchain_core.embeddings import Embeddings
from interfaces.singleton import SingeltonLayer


class EmbeddingInterface(metaclass=SingeltonLayer):

    @property
    @abstractmethod
    def get_client() -> Embeddings:
        pass