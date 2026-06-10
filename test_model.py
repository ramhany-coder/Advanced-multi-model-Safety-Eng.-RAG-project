from langchain_community.embeddings import HuggingFaceEmbeddings

print("start")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

print("created")