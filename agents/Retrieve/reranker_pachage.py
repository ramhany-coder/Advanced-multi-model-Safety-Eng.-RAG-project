from langchain_core.documents import Document
from config import settings

class Pinecone :
    def __init__(self,model:str,k:int): 
        try:
            from langchain_pinecone import PineconeRerank
        except Exception as e :
            PineconeRerank = None
            raise ValueError(f"Error in loading Pinecone reranker model :{str(e)}")
        
        self.reranker = PineconeRerank(
                    model = model, 
                    top_k = k
                )

    def rerank(self,documents, query: str):
        if self.reranker is not None :
            try : 
                return self.reranker.compress_documents(
                            documents=documents,
                            query=query)
            except Exception as e :
                raise ValueError(f"Error during the reranking step by Pinecone : {str(e)}")
        

pinecone_client = Pinecone(settings.PINECONE_MODEL,8)