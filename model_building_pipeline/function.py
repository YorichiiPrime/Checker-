'''this script inspects the weak labels assigned to each line in a specified PDF.
It loads the weak labels from a JSON file and prints each line along with its assigned label.
it helps to check the weak label json and pdf pdf line by line heading with line... help to edit the json to make it good one.

this help to make model training good'''
import json
import os
from insights import main_ex
from extractor import extract_document_lines

# -----------------------------
# CONFIG
# -----------------------------

PDF_PATH = "modularity\\pdfs\\doc8.pdf"
LABELS_JSON = "weak_labels_final.json"


# -----------------------------
# LOAD LABELS
# -----------------------------

with open(LABELS_JSON, "r", encoding="utf-8") as f:
    ALL_LABELS = json.load(f)

pdf_name = os.path.basename(PDF_PATH)
pdf_labels = ALL_LABELS.get(pdf_name, {})


# -----------------------------
# PRINT LINE BY LINE
# -----------------------------

lines = extract_document_lines(PDF_PATH)

print(f"\n[+] Inspecting labels for: {pdf_name}\n")

for line in lines:
    key = f"{line['page_index']}_{line['line_index']}"
    label = pdf_labels.get(key, "UNLABELED")

    print(
        f"[Page {line['page_index']:>2} | Line {line['line_index']:>3}] "
        f"{label:<10} :: {line['text']}"
    )

print("\n[✓] Inspection complete")

