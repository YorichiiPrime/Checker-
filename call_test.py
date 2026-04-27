
from rag_engine.converters.structuring_json import create_structured_json
from rag_engine.converters.vector_build import build_vector_db
import os
from docx2pdf import convert

import time
def convert_docx_to_pdf(docx_path):

    pdf_path = docx_path.replace(".docx", ".pdf")
    convert(docx_path, pdf_path)

    return pdf_path


def process_root_directory():

    root_dir = os.getcwd()

    for file in os.listdir(root_dir):

        file_path = os.path.join(root_dir, file)

        if file.endswith(".pdf"):

            print("PDF detected:", file)

            create_structured_json(file_path)

            time.sleep(3)

            build_vector_db("structured.json")


        elif file.endswith(".docx"):

            print("DOCX detected:", file)

            pdf_path = convert_docx_to_pdf(file_path)

            create_structured_json(pdf_path)

            time.sleep(3)

            build_vector_db("structured.json")

    print("Processing complete.")

process_root_directory()

# ----------------------------
# Continuous Test Loop
# ----------------------------
'''
if __name__ == "__main__":
    from rag_engine.vector_search import fast_search
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
            print(f"Content: {doc[:500]}...") # Showing first 200 chars
            print("-" * 100)'''
