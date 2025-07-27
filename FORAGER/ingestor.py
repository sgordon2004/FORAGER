"""
This module handles the ingestion of raw documents into the FORAGER pipeline by extracting clean, structured text 
from various file formats, including PDFs and HTML files. It converts messy, unstructured source documents into 
plain text files and accompanying JSON metadata, making them suitable for downstream chunking, embedding, and retrieval.

Core Functionality:
- Extracts text from PDF documents using pdfplumber, with optional tolerance adjustments for improved accuracy.
- Extracts visible, main content from HTML files using a combination of heuristic block selection and the Readability algorithm.
- Cleans and normalizes extracted text, removing unnecessary whitespace and formatting.
- Saves the processed text into dedicated text directories and generates JSON metadata files containing document statistics.

Key Features:
- Supports batch processing of all documents in the FORAGER corpus input directories.
- Automatically creates required directories for storing extracted text and metadata.
- Designed for extensibility to support additional document formats (e.g., Markdown) if needed.

Intended Usage:
Run this module to extract and clean all PDF and HTML documents in the FORAGER corpus. The resulting plain text and metadata 
files serve as the foundation for the retrieval-augmented generation (RAG) pipeline by enabling efficient indexing and retrieval.
"""

import pdfplumber
from bs4 import BeautifulSoup as bs
from pathlib import Path
from readability import Document
import json
__docformat__ = "google"
base = Path("FORAGER_corpus/heterogenous_integration") # Base directory for the FORAGER corpus's HI information
pdf_dir = base / "pdf" # Sub-directory containing PDF files
html_dir = base / "html" # Sub-directory containing HTML files
pdf_text_dir = base / "pdf_text" # Sub-directory containing extracted PDF text
html_text_dir = base / "html_text" # Sub-directory containing extracted HTML text
json_dir = base / "json"

# Create directory to store extracted text if it doesn't exist
pdf_text_dir.mkdir(parents=True, exist_ok=True)
html_text_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Extract raw text for each document in the FORAGER corpus
# Extract text from PDFs

def extract_pdf(filename, x_tolerance=2.0):
    """
    Extracts text from a single PDF file using pdfplumber.

    Args:
        filename (str): The name of the PDF file to extract text from.
        x_tolerance (float): The horizontal tolerance for text extraction. Default is 2.0.
    
    Returns:
        str: Extracted text from the PDF file.
    """
    pdf_path = pdf_dir / filename
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text(x_tolerance=x_tolerance) + "\n"
        print(f"Extracted text from {filename} with x_tolerance={x_tolerance}")
    return text

def dump_pdf_text(filename, text):
    """
    Dumps the extracted text to a file in the pdf_text directory and a .json metadata file.

    Args:
        text (str): The extracted text to be saved.
    """
    filepath = pdf_text_dir / f"{Path(filename).stem}.txt"
    # with open(filepath, 'w', encoding="utf-8") as f:
    #     f.write(text)
    # print(f"Extracted text saved to {filepath}")

    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    # Save .json
    json_output = {
        "source_filename": filepath.name,
        "title": Path(filepath).stem,
        "content": cleaned,
        "content_length": len(cleaned),
        "num_paragraphs": cleaned.count("\n")
    }

    json_dir.mkdir(exist_ok=True)
    json_path = json_dir / (Path(filename).stem + ".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=2)
    print(f"[OK] Saved text and JSON for {filename}")

def extract_all_pdfs(x_tolerance=2.0):
    """
    Extracts text from all PDF files in the specified directory using pdfplumber.

    Args:
        x_tolerance (float): The horizontal tolerance for text extraction. Default is 2.0.
    """
    for file_path in pdf_dir.glob("*.pdf"):
            # Save or process the extracted text as needed
            text = extract_pdf(file_path.name, x_tolerance=x_tolerance)
            dump_pdf_text(file_path.name, text)



# HTML

def extract_structured_blocks(soup):
    known_classes = [
        "cmp-text__block",
        "cmp-text",
        "cmp-container",
        "responsivegrid"
    ]
    
    content_blocks = []
    for cls in known_classes:
        blocks = soup.find_all("div", class_=cls)
        content_blocks.extend(blocks)

    # Optional: filter only blocks with paragraph tags
    content_blocks = [block for block in content_blocks if block.find("p")]

    if content_blocks:
        return bs("".join(str(b) for b in content_blocks), "html.parser")
    
    return None

def clean_html(input_path: Path, output_dir: Path, json_dir: Path) -> Path:
    """
    Extracts visible, cleaned text from an HTML file and saves it as a .txt file.

    Args:
        input_path (Path): Path to the input HTML file.
        output_dir (Path): Directory to save the cleaned text file.
        json_dir (Path): Directory to save the metadata JSON file.

    Returns:
        Path: Path to the saved .txt file.
    """
    html = input_path.read_text(encoding='utf-8')
    soup = bs(html, 'html.parser')

    # Heuristic: ASE-style multi-block layout
    structured = extract_structured_blocks(soup)
    if structured:
        print(f"[STRUCTURED] Using multi-block fallback for {input_path.name}")
        content_soup = structured
        title = soup.title.get_text(strip=True) if soup.title else input_path.stem
    else:
        # Use readability to isolate the site's main content (avoids header, navbar, etc.)
        try:
            doc = Document(html)
            title = doc.short_title()
            cleaned_html = doc.summary()
            content_soup = bs(cleaned_html, 'html.parser')
        except Exception as e:
            print(f"[ERROR] Readability failed on {input_path.name}: {e}")
            return None

    # Convert headers and links to Markdown
    for h in content_soup.find_all(['h1', 'h2', 'h3']): # Convert headers to Markdown
        level = int(h.name[1])
        h.insert_before(f"\n{'#' * level} {h.get_text(strip=True)}\n")

    for a in content_soup.find_all('a', href=True): # Convert link anchors to Markdown
        a.replace_with(f"{a.get_text(strip=True)} ({a['href']})")

    # Clean and save text
    text = content_soup.get_text(separator='\n')
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    if not cleaned:
        print(f"[WARNING] Empty cleaned text for {input_path.name}")
        return None

    output_path = output_dir / (input_path.stem + ".txt")
    output_dir.mkdir(exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(cleaned)

    # Save .json
    json_output = {
        "source_filename": input_path.name,
        "title": title,
        "content": cleaned,
        "content_length": len(cleaned),
        "num_paragraphs": cleaned.count("\n")
    }

    json_dir.mkdir(exist_ok=True)
    json_path = json_dir / (input_path.stem + ".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=2)
    print(f"[OK] Saved text and JSON for {input_path.name}")
    
    print(f"[OK] Readability extracted {input_path.name} with title '{title}'")
    return output_path

def extract_all_html():
    """
    Extracts text from all HTML files in the specified directory using BeautifulSoup.
    """
    for html_path in html_dir.glob("*.html"):
        output_path = clean_html(html_path, html_text_dir, json_dir)
        print(f"Cleaned HTML saved to {output_path}")