"""
Weak Label Generator (FINAL – Binary Version)

Label space:
- HEADING
- PARAGRAPH

Philosophy:
1. First, aggressively identify PARAGRAPHS (hard rules)
2. Only remaining short, non-sentence lines can be HEADING
3. HEADING requires clear visual dominance (relative size)

This generates SILVER labels.
DO NOT use for evaluation.
"""

import json
import os
from statistics import median
from insights import main_ex
from extractor import extract_document_lines


# ------------------------------
# Utilities
# ------------------------------

def line_font_size(line):
    """
    Returns median font size of the line (or None)
    """
    sizes = []
    for size, count in line["size_stats"].items():
        sizes.extend([size] * count)
    return median(sizes) if sizes else None


# ------------------------------
# PHASE 1: PARAGRAPH GATE
# ------------------------------

def is_paragraph(line, insights):
    wc = line["word_count"]

    # Rule 1: long text is paragraph
    if wc >= 12:
        return True

    # Rule 2: sentence-like lines
    if wc >= 8 and line["ends_with_punctuation"]:
        return True

    # Rule 3: dominant paragraph style + reasonable length
    if (
        insights.get("paragraph_font") in line["style_stats"]
        and insights.get("paragraph_size") in line["size_stats"]
        and wc >= 6
    ):
        return True

    return False


# ------------------------------
# PHASE 2: HEADING DECISION
# ------------------------------

def is_heading(line, insights):
    wc = line["word_count"]
    size = line_font_size(line)

    # Must be short
    if wc > 6:
        return False

    # Must NOT look like a sentence
    if line["ends_with_punctuation"]:
        return False

    # Must be visually distinct by SIZE (not font name, not bold)
    if (
        size
        and insights.get("paragraph_size")
        and size > insights["paragraph_size"]
    ):
        return True

    return False


# ------------------------------
# MAIN PIPELINE
# ------------------------------

def generate_weak_labels(pdf_paths, output_path="weak_labels_final.json"):
    all_labels = {}

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        print(f"[+] Processing {pdf_name}")

        lines = extract_document_lines(pdf_path)
        insights = main_ex(pdf_path)

        pdf_labels = {}

        for line in lines:
            key = f"{line['page_index']}_{line['line_index']}"

            # Phase 1: Paragraph gate
            if is_paragraph(line, insights):
                label = "PARAGRAPH"
            else:
                # Phase 2: Heading or fallback
                label = "HEADING" if is_heading(line, insights) else "PARAGRAPH"

            pdf_labels[key] = label

        all_labels[pdf_name] = pdf_labels

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2)

    print(f"[✓] Weak labels saved to {output_path}")



# Example usage:
pdfs = [
    "modularity\\pdfs\\doc1.pdf",
    "modularity\\pdfs\\doc2.pdf",
    "modularity\\pdfs\\doc3.pdf",
    "modularity\\pdfs\\doc4.pdf",
    "modularity\\pdfs\\doc5.pdf",
    "modularity\\pdfs\\doc6.pdf",
    "modularity\\pdfs\\doc7.pdf",
    "modularity\\pdfs\\doc8.pdf"
]

generate_weak_labels(pdfs)

