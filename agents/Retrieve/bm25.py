from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25:
    def __init__(
        self,
        documents: list[Document],
        k: int = 5,
    ):
        if not documents:
            raise ValueError(
                "Cannot create BM25 retriever from empty documents."
            )

        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = k

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)