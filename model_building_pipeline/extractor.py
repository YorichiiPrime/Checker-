import pdfplumber
from docx import Document
import re
import os

from wordsegment import load, segment
import re

load()

def repair_sentence(all_lines):
    for line in all_lines:
        text = line["text"]
        text = re.sub(r'\(cid:\d+\)', ' ', text)
        # 1. First, handle CamelCase to prevent "Longformer" becoming "long former"
        # We add a space before capital letters that follow a lowercase letter
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        
        # 2. Split the string by actual spaces (if any exist)
        parts = text.split()
        
        final_repaired = []

        for part in parts:
            # 3. Use regex to find all "chunks" of letters or non-letters
            # This keeps "architecture," as ["architecture", ","]
            chunks = re.findall(r'[A-Za-z]+|[^A-Za-z]+', part)
            
            repaired_chunks = []
            for chunk in chunks:
                # If the chunk is purely alphabetic and long, segment it
                if chunk.isalpha() and len(chunk) > 3:
                    # segment() returns a list of words
                    split_words = segment(chunk)
                    
                    # Heuristic: If segmenting didn't change anything, keep original casing
                    # Otherwise, use the segmented version
                    if len(split_words) == 1 and split_words[0].lower() == chunk.lower():
                        repaired_chunks.append(chunk)
                    else:
                        repaired_chunks.append(" ".join(split_words))
                else:
                    repaired_chunks.append(chunk)
            
            final_repaired.append("".join(repaired_chunks))

        final_repaired = " ".join(final_repaired)
        
        chars = len(final_repaired)
        alpha = sum(c.isalpha() for c in final_repaired)
        digits = sum(c.isdigit() for c in final_repaired)
        symbols = sum((not c.isalnum() and not c.isspace()) for c in final_repaired)
        math_symbols = set("=<>+-*/^θλβˆ∑≈≤≥")
        math_symbol_count = sum(c in math_symbols for c in final_repaired)

        word_count = len(final_repaired.split())

        
        line["text"] = final_repaired
        line["word_count"] = word_count

        # Tiny line (very short fragments)
        line["is_tiny"] = 1 if word_count <= 2 else 0

        # Pure number line (e.g., "1", "2")
        line["is_numeric_only"] = 1 if final_repaired.strip().isdigit() else 0

        # Character ratios
        line["alpha_ratio"] = alpha / chars if chars else 0
        line["digit_ratio"] = digits / chars if chars else 0
        line["symbol_ratio"] = symbols / chars if chars else 0

        # Contains math symbols
        line["has_math_symbol"] = 1 if math_symbol_count > 0 else 0

    return all_lines

# ==========================================================
# COMMON HELPERS
# ==========================================================

def build_stats(items):
    stats = {}
    for item in items:
        stats[item] = stats.get(item, 0) + 1
    return stats


def has_symbol(text):
    return bool(re.search(r"[•●▪■→]", text))


def starts_with_number(text):
    return bool(re.match(r"^\d+[\.\)]", text.strip()))


def ends_with_punctuation(text):
    return text.strip().endswith((".", "!", "?"))


# ==========================================================
# PDF EXTRACTOR (YOUR ORIGINAL LOGIC)
# ==========================================================

#Extract lines from pdf along with their attributes and stats
def extract_lines_pdf(words, page_index, start_line_index, threshold=2):
    lines = []

    current_text = ""
    current_top = None
    current_fonts = []
    current_sizes = []

    line_index = start_line_index

    for word in words:
        word_text = word["text"]
        word_top = word["top"]
        word_font = word["fontname"]
        word_size = word["size"]

        if current_top is None:
            current_top = word_top
            current_text = word_text
            current_fonts = [word_font]
            current_sizes = [word_size]

        elif abs(word_top - current_top) <= threshold:
            current_text += " " + word_text

            if word_font not in current_fonts:
                current_fonts.append(word_font)

            if word_size not in current_sizes:
                current_sizes.append(word_size)

        else:
            lines.append({
                "text": current_text,
                "line_index": line_index,
                "page_index": page_index,
                "layout": {"top": current_top},
                "word_count": len(current_text.split()),
                "size_stats": build_stats(current_sizes),
                "style_stats": build_stats(current_fonts),
                "has_symbol": has_symbol(current_text),
                "starts_with_number": starts_with_number(current_text),
                "ends_with_punctuation": ends_with_punctuation(current_text)
            })

            line_index += 1

            current_top = word_top
            current_text = word_text
            current_fonts = [word_font]
            current_sizes = [word_size]

    if current_text:
        lines.append({
            "text": current_text,
            "line_index": line_index,
            "page_index": page_index,
            "layout": {"top": current_top},
            "word_count": len(current_text.split()),
            "size_stats": build_stats(current_sizes),
            "style_stats": build_stats(current_fonts),
            "has_symbol": has_symbol(current_text),
            "starts_with_number": starts_with_number(current_text),
            "ends_with_punctuation": ends_with_punctuation(current_text)
        })

        line_index += 1

    return lines, line_index


def extract_pdf_lines(pdf_path):
    all_lines = []
    current_line_index = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["size", "fontname"])

            page_lines, current_line_index = extract_lines_pdf(
                words,
                page_index,
                current_line_index
            )

            all_lines.extend(page_lines)
    all_lines = repair_sentence(all_lines)

    return all_lines


# ==========================================================
# DOCX EXTRACTOR
# ==========================================================

def extract_docx_lines(docx_path):
    document = Document(docx_path)

    all_lines = []
    line_index = 0

    for para_index, paragraph in enumerate(document.paragraphs):

        text = paragraph.text.strip()
        if not text:
            continue

        fonts = []
        sizes = []

        for run in paragraph.runs:
            if run.text.strip():
                font_name = run.font.name if run.font.name else "Unknown"
                fonts.append(font_name)

                if run.font.size:
                    sizes.append(round(run.font.size.pt, 1))
                else:
                    sizes.append(0)

        if not sizes:
            sizes = [0]

        all_lines.append(build_line_dict(
            text,
            line_index,
            page_index=0,
            top=para_index,      # simulated top
            sizes=sizes,
            fonts=fonts
        ))

        line_index += 1

    return all_lines


# ==========================================================
# SHARED LINE DICT BUILDER
# ==========================================================

def build_line_dict(text, line_index, page_index, top, sizes, fonts):
    return {
        "text": text,
        "line_index": line_index,
        "page_index": page_index,
        "layout": {"top": top},
        "word_count": len(text.split()),
        "size_stats": build_stats(sizes),
        "style_stats": build_stats(fonts),
        "has_symbol": has_symbol(text),
        "starts_with_number": starts_with_number(text),
        "ends_with_punctuation": ends_with_punctuation(text)
    }


# ==========================================================
# UNIFIED ENTRY POINT
# ==========================================================

def extract_document_lines(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf_lines(file_path)

    elif ext == ".docx":
        return extract_docx_lines(file_path)

    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX allowed.")
