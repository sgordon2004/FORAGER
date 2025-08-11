import pymupdf
from pathlib import Path
import re

base = Path("FORAGER_corpus/heterogenous_integration") # Base directory for the FORAGER corpus's HI information
pdf_dir = base / "pdf" # Sub-directory containing PDF files
alt_pdf_dir = base / "pdfs" # Sub-directory PDFs are in when uploaded via Streamlit
pdf_text_dir = base / "pdf_text"
pdf_text_dir.mkdir(parents=True, exist_ok=True)

def extract_pdf(filename):
    file = pdf_dir / filename if (pdf_dir / filename).exists() else (alt_pdf_dir / filename)
    output_file = pdf_text_dir
    file_name_with_ext = file.name
    file_stem = Path(file_name_with_ext).stem # Name without extension
    doc = pymupdf.open(file)

    full_text = ""
    with open(output_file / f"{file_stem}.txt", "w", encoding="utf-8") as f:
        for page_num, page in enumerate(doc):
            page_height = page.rect.height
            page_width = page.rect.width
            blocks = page.get_text("blocks")
            if not blocks:
                continue
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            # f.write(f"\n--- Page {page_num + 1} ---\n\n")
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type, *_ = block
                text_lower = text.lower().strip()
                if y0 < 0.06 * page_height or y1 > 0.92 * page_height:
                    continue
                if block_type != 0:
                    continue
                if text_lower.startswith(("fig.", "figure", "table", "downloaded from")):
                    continue
                cleaned_text = text.strip()
                f.write(cleaned_text + "\n")
                full_text += cleaned_text + "\n"

    return full_text

def extract_all_pdfs():
    for file_path in pdf_dir.glob("*.pdf"):
        extract_pdf(file_path.name)