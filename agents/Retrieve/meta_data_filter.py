from langchain_core.documents import Document
import json


class MetadataFilter:

    def __init__(self, documents_path: str):
        self.documents_path = documents_path

    def load_documents(self) -> list[dict]:
        with open(self.documents_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _to_document(doc: dict) -> Document:
        return Document(
            page_content=doc.get("full_text") or doc.get("extractive_summary") or "",
            metadata={
                "doc_id": doc.get("doc_id", ""),
                "section_id": doc.get("section_id"),
                "title": doc.get("title", ""),
                "url": doc.get("url") or doc.get("source") or "",
            },
        )

    def get_results(self, section_ids: list[str]) -> list[Document]:
        docs = self.load_documents()
        return [
            self._to_document(doc)
            for doc in docs
            if doc.get("section_id") in section_ids
        ]

    def get_all(self) -> list[Document]:
        return [self._to_document(doc) for doc in self.load_documents()]