import joblib
from .insights import main_ex
from .extractor import extract_document_lines

MODEL_PATH = "heading_classifier.joblib"
PDF_PATH = "test.pdf"   # change this to the PDF you want to test

# LOAD MODEL

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
label_map = artifact["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

# FEATURE EXTRACTION

def line_to_features(line, insights):
    fonts = set(line["style_stats"].keys())
    sizes = set(line["size_stats"].keys())

    return [
        line["word_count"],
        int(line["has_symbol"]),
        int(line["starts_with_number"]),
        int(line["ends_with_punctuation"]),
        int(insights.get("paragraph_font") in fonts),
        int(insights.get("heading_size") in sizes if insights.get("heading_size") else 0),
        round(line["layout"]["top"] / 800, 3),

        line.get("is_tiny", 0),
        line.get("is_numeric_only", 0),
        round(line.get("alpha_ratio", 0), 3),
        round(line.get("digit_ratio", 0), 3),
        round(line.get("symbol_ratio", 0), 3),
        line.get("has_math_symbol", 0),
    ]


# RUN TEST

def classify_pdf(pdf_path):
    lines = extract_document_lines(pdf_path)
    insights = main_ex(lines)
    f_matrix = [line_to_features(line, insights) for line in lines]
    pred = model.predict(f_matrix)

    structured_output = []

    for i, line in enumerate(lines):
        
        label = inv_label_map[pred[i]]

        structured_output.append({
            "page_index": line["page_index"],
            "line_index": line["line_index"],
            "label": label,
            "text": line["text"]
        })

    return structured_output

# Example usage in case to test the classify_pdf function directly without running the whole server
#remove the triple quotes to run this test
'''
if __name__ == "__main__":
    print(f"\n[+] Model Classify on: {PDF_PATH}\n")

    results = classify_pdf(PDF_PATH)

    for item in results:
        print(
            f"[Page {item['page_index']:>2} | Line {item['line_index']:>3}] "
            f"{item['label']:<9} :: {item['text']}"
        )

    print("\n[✓] Test complete")'''
