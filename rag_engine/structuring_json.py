
from rag_engine.converters.extract_classify.classify_model import classify_pdf
import json


def create_structured_json(pdf_path, output_path="structured.json", max_words=350):
    classified_lines = classify_pdf(pdf_path)

    structured = []
    current_section = None

    for item in classified_lines:
        label = item["label"]
        text = item["text"].strip()

        if not text:
            continue

        # HEADING LOGIC
        if label == "HEADING":

            # Skip tiny garbage headings (like "n", "i")
            if len(text.split()) == 1 and len(text) <= 3:
                continue

            # Merge consecutive headings (if needed)
            if current_section and current_section["type"] == "heading":
                if not text[0].isdigit():
                    current_section["heading"] += " " + text
                    continue

            # Start new section
            current_section = {
                "type": "heading",
                "heading": text,
                "content": []
            }

            structured.append(current_section)

        # PARAGRAPH LOGIC
        elif label == "PARAGRAPH":

            if current_section is None:
                current_section = {
                    "type": "heading",
                    "heading": "INTRO",
                    "content": []
                }
                structured.append(current_section)

            current_section["content"].append(text)

    # CHUNK SPLITTING BY SIZE
    final_output = []

    for section in structured:
        if not section["content"]:
            continue

        # Instead of " ".join(), we rebuild the text based on your "glue hints"
        full_text = ""
        lines = section["content"]
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if line.endswith("-"):
                # Remove the hyphen and add the line WITHOUT a space
                full_text += line[:-1]
            else:
                # Add the line with a space (standard paragraph behavior)
                full_text += line + (" " if idx < len(lines) - 1 else "")
        
        full_text = full_text.strip()
        #END OF IMPROVED JOINING 

        words = full_text.split()
        chunks = []
        chunk_id = 0

        for i in range(0, len(words), max_words):
            chunk_text = " ".join(words[i:i + max_words])

            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_text
            })
            chunk_id += 1

        final_output.append({
            "heading": section["heading"],
            "chunks": chunks
        })

    # SAVE JSON

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    return final_output
