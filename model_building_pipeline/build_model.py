"""
Binary Heading vs Paragraph Model Training

Label space:
0 -> PARAGRAPH
1 -> HEADING

Uses:
- check.py (extract_pdf_lines, main_ex)
- weak_labels_final.json (SILVER labels)

NOTE:
- Evaluation here is sanity-check only
- Real evaluation must be done on manually labeled PDFs
"""

import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from modularity.converters.extract_classify.insights import main_ex
from modularity.converters.extract_classify.extractor import extract_document_lines


# -----------------------------
# CONFIG
# -----------------------------

PDF_PATHS = [
    "modularity\\pdfs\\doc1.pdf",
    "modularity\\pdfs\\doc2.pdf",
    "modularity\\pdfs\\doc3.pdf",
    "modularity\\pdfs\\doc4.pdf",
    "modularity\\pdfs\\doc5.pdf",
    "modularity\\pdfs\\doc6.pdf",
    "modularity\\pdfs\\doc7.pdf",
    "modularity\\pdfs\\doc8.pdf",
]

LABELS_PATH = "weak_labels_final.json"
MODEL_OUT = "heading_classifier.joblib"


LABEL_MAP = {
    "PARAGRAPH": 0,
    "HEADING": 1
}


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def line_to_features(line, insights):
    fonts = set(line["style_stats"].keys())
    sizes = set(line["size_stats"].keys())

    features = [
        line["word_count"],
        int(line["has_symbol"]),
        int(line["starts_with_number"]),
        int(line["ends_with_punctuation"]),
        int(insights.get("paragraph_font") in fonts),
        int(insights.get("heading_size") in sizes if insights.get("heading_size") else 0),
        round(line["layout"]["top"] / 800, 3),  # normalized vertical position

        line.get("is_tiny", 0),
        line.get("is_numeric_only", 0),
        round(line.get("alpha_ratio", 0), 3),
        round(line.get("digit_ratio", 0), 3),
        round(line.get("symbol_ratio", 0), 3),
        line.get("has_math_symbol", 0),
    ]

    return features


# -----------------------------
# LOAD LABELS
# -----------------------------

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    ALL_LABELS = json.load(f)


# -----------------------------
# BUILD DATASET
# -----------------------------

X = []
y = []

print("[+] Building dataset...")

for pdf_path in PDF_PATHS:
    pdf_name = os.path.basename(pdf_path)
    print(f"    Processing {pdf_name}")

    lines = extract_document_lines(pdf_path)
    insights = main_ex(lines)

    pdf_labels = ALL_LABELS.get(pdf_name, {})

    for line in lines:
        key = f"{line['page_index']}_{line['line_index']}"

        if key not in pdf_labels:
            continue  # skip unlabeled lines

        label_str = pdf_labels[key]

        # Safety check
        if label_str not in LABEL_MAP:
            continue

        X.append(line_to_features(line, insights))
        y.append(LABEL_MAP[label_str])

print(f"[✓] Total labeled samples: {len(y)}")

if len(y) < 100:
    print("[!] WARNING: Very small dataset — model may be unstable")


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -----------------------------
# TRAIN MODEL
# -----------------------------

print("[+] Training model...")

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    min_samples_split=5,
    oob_score=True,
    class_weight='balanced'
)

model.fit(X_train, y_train)


# -----------------------------
# SANITY EVALUATION (NOT FINAL)
# -----------------------------

y_pred = model.predict(X_test)

print("\nConfusion Matrix (SANITY CHECK):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (SANITY CHECK):")
print(classification_report(y_test, y_pred, target_names=["PARAGRAPH", "HEADING"]))


# -----------------------------
# SAVE MODEL + METADATA
# -----------------------------

artifact = {
    "model": model,
    "label_map": LABEL_MAP,
    "feature_order": [
        "word_count",
        "has_symbol",
        "starts_with_number",
        "ends_with_punctuation",
        "is_paragraph_font",
        "is_heading_size",
        "normalized_top",
        "is_tiny",
        "is_numeric_only",
        "alpha_ratio",
        "digit_ratio",
        "symbol_ratio",
        "has_math_symbol"
    ]
}

joblib.dump(artifact, MODEL_OUT)

print(f"\n[✓] Model saved as {MODEL_OUT}")
