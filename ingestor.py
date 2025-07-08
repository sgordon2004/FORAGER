"""
This module extracts raw text from documents within the FORAGER corpus.
It converts all source documents (PDF, HTML, Markdown, etc.) into clean, structured plain text chunks, enriched with metadata,
and prepared for embedding and retrieval.
It turns messy raw content into queryable knowledge.
"""

import pdfplumber
import os

base = "FORAGER_corpus/heterogenous_integration" # Base directory for the FORAGER corpus's HI information
pdf_dir = os.path.join(base, "pdf") # Sub-directory containing PDF files
html_dir = os.path.join(base, "html") # Sub-directory containing HTML files

# Create directory to store extracted text if it doesn't exist
text_dir = os.path.join(base, "text")
os.makedirs(text_dir, exist_ok=True)

# Step 1: Extract raw text for each document in the FORAGER corpus
# Extract text from PDFs
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        with pdfplumber.open(pdf_path) as pdf:
            name = os.path.join(pdf_dir, filename)
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            text = ""
            for page in pdf.pages:
                text += page.extract_text(x_tolerance = 2.0) + "\n"
        # Save or process the extracted text as needed
        print(f"Extracted text from {filename}:\n{text[:100]}...")  # Print first 100 characters for brevity
        filepath = os.path.join(text_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(filepath, 'w') as f:
            f.write(text)

# Trying fitz

# HTML



# Step 2: Generate or update metadata

# Step 3: Chunk the text (~800 word chunks)

# Step 4: Embed chunks