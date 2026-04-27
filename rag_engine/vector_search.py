import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import time

from rag_engine.converters.vector_build import MODEL_NAME, COLLECTION_NAME, DB_PATH


print("--- Loading Resources into RAM ---")
start_load = time.time()

# Load model once globally
model = SentenceTransformer(MODEL_NAME)

# Load Chroma once globally (PersistentClient caches the index in RAM while running)
client = PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

print(f"--- Ready! (Loaded in {time.time() - start_load:.2f}s) ---")

def fast_search(query, top_k=3):
    """Search function using the globally loaded model and collection."""
    start_time = time.time()
    
    # 1. Encode query
    query_embedding = model.encode(query).tolist()
    
    # 2. Query vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    end_time = time.time()
    return results, (end_time - start_time)


