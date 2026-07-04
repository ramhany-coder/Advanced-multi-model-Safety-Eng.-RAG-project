from bm25_retriever import BM25Retriever

def test_bm25_retriever():
    documents = [
        {"text": "The cat sat on the mat."},
        {"text": "The dog sat on the log."},
        {"text": "The cat and the dog are friends."}
    ]
    retriever = BM25Retriever(documents)
    
    query = "cat mat"
    results = retriever.retrieve(query, k=2)
    
    assert len(results) == 2
    assert results[0]["text"] == "The cat sat on the mat."
    assert results[1]["text"] == "The cat and the dog are friends."
    print("BM25Retriever test passed.")
