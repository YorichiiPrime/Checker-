import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import time


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"
DB_PATH = "chroma_db"

model = SentenceTransformer(MODEL_NAME)

# Build Vector Database

def build_vector_db(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Clear old data
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    print("Preparing data...")

    all_texts = []
    all_ids = []
    all_metadatas = []

    for idx, section in enumerate(data):

        heading = section.get("heading", "UNKNOWN")
        chunks = section.get("chunks", [])

        for chunk_idx, chunk in enumerate(chunks):

            if isinstance(chunk, dict):
                # CHANGE THIS LINE: from "text" to "content"
                chunk_text = chunk.get("content", "") 
            else:
                chunk_text = chunk

            if not isinstance(chunk_text, str) or not chunk_text.strip():
                continue

            combined_text = f"Section: {heading}\nContent: {chunk_text}"
            all_texts.append(combined_text)
            all_ids.append(f"{idx}_{chunk_idx}")
            all_metadatas.append({"heading": heading})

    print(f"Encoding {len(all_texts)} chunks...")
    
    if len(all_texts) == 0:
        print("No valid text chunks found. Aborting build.")
        return 

    embeddings = model.encode(
        all_texts,
        batch_size=32,
        show_progress_bar=True
    )

    print("Storing in Chroma...")

    collection.add(
        ids=all_ids,
        embeddings=embeddings.tolist(),
        documents=all_texts,
        metadatas=all_metadatas
    )

    print("Vector DB built successfully.")
    print("Total items in DB:", collection.count())




'''
# ----------------------------
# Continuous Test Loop
# ----------------------------
if __name__ == "__main__":
    print("\n[Vektor RAG Speed Tester]")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_query = input("\nEnter Search Query: ").strip()
        
        if user_query.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        if not user_query:
            continue

        # Perform Search
        results, duration = fast_search(user_query)

        print(f"Found {len(results['documents'][0])} results in {duration:.4f} seconds.")
        print("-" * 30)

        # Print results with metadata
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            
            print(f"RANK {i+1} | Score: {dist:.4f} | Section: {meta['heading']}")
            print(f"Content: {doc[:400]}...") # Showing first 200 chars
            print("-" * 10)'''