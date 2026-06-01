import sys
import os
import json
import uuid

# --- METAPROGRAMMING PATCH FOR LANGCHAIN 0.3+ COMPATIBILITY ---
# This explicitly tricks older third-party modules looking for 'langchain.retrievers'
import langchain_community.retrievers as community_retrievers
sys.modules['langchain.retrievers'] = community_retrievers
# --------------------------------------------------------------

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Now this import will load flawlessly because its dependency path is aliased above!
from bm25_retriever.retriever import PersistentBM25Retriever

def load_scraped_data(json_path):
    """Loads the raw scraped OSHA JSON file and turns it into LangChain Documents."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Did you run the scraper first?")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    documents = []
    for item in raw_data:
        doc = Document(
            page_content=item['full_text'],
            metadata={
                "source": item['url'],
                "section_id": item['section_id'],
                "title": item['title']
            }
        )
        documents.append(doc)
        
    print(f"Loaded {len(documents)} source compliance documents from JSON.")
    return documents

def main():
    # 1. Splitter Strategy Configurations
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 2. Production Embeddings & Dense Store (Chroma)
    print("Initializing production HuggingFace Embedding Engine...")
    production_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Connecting to persistent Vector Store (Chroma)...")
    vectorstore = Chroma(
        collection_name="production_parent_child_store", 
        embedding_function=production_embeddings,
        persist_directory="osha"  # Directly matches your agent's loading lookups
    )

    # 3. Target Raw Source Path
    raw_json_file = "osha_raw_documents.json"
    backup_json_path = "./parent_doc_store_backup.json"
    
    try:
        docs = load_scraped_data(raw_json_file)
        
        child_docs_to_embed = []
        parent_store_map = {}
        
        print("\nManually calculating Parent-Child structural mapping layout...")
        for doc in docs:
            # Chunk into larger Parent sections
            parents = parent_splitter.split_documents([doc])
            
            for parent in parents:
                parent_id = str(uuid.uuid4())
                
                # Cache the parent structure
                parent_store_map[parent_id] = {
                    "page_content": parent.page_content,
                    "metadata": parent.metadata
                }
                
                # Split this specific parent down into smaller children chunks
                children = child_splitter.split_documents([parent])
                for child in children:
                    # Inject parent tracking pointer directly into child metadata mapping
                    child.metadata["doc_id"] = parent_id
                    child_docs_to_embed.append(child)

        # 4. Ingest Dense Vectors
        print(f"Adding {len(child_docs_to_embed)} child vector nodes into Chroma ('osha')...")
        vectorstore.add_documents(child_docs_to_embed)
        
        # 5. Save the Parent text pointers out to disk
        print(f"Persisting parent documents mapping dictionary to disk...")
        with open(backup_json_path, 'w', encoding='utf-8') as f:
            json.dump(parent_store_map, f, ensure_ascii=False, indent=4)

        # 6. Generate the Sparse Matrix for Hybrid Processing
        print("\nBuilding Sparse Index (PersistentBM25Retriever) for Ensemble Routing...")
        sparse_dir = "osha_sparse"
        os.makedirs(sparse_dir, exist_ok=True)
        
        bm25_retriever = PersistentBM25Retriever.from_documents(
            documents=docs, 
            save_dir=sparse_dir
        )
        print(f"SUCCESS: Sparse Index populated and saved to: ./{sparse_dir}")

        print("\n--- Ingestion Pipeline Completed Successfully! ---")
        print(f"-> Dense vectors stored safely in: ./osha")
        print(f"-> Sparse matrix stored safely in: ./{sparse_dir}")
        print(f"-> Parent mappings cached locally in: {backup_json_path}")

    except FileNotFoundError as e:
        print(f"\n[Error executing pipeline]: {e}")

if __name__ == "__main__":
    main()