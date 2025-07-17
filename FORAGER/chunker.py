"""
This module processes a collection of JSON documents, tokenizes their content, 
and splits the text into overlapping chunks for downstream embedding and retrieval tasks 
within the FORAGER pipeline.

Core Functionality:
- Loads JSON files from a specified input directory, each containing a document's full text.
- Tokenizes the document content using the Llama 3 tokenizer to ensure consistency with LLM inference.
- Splits the tokenized content into fixed-size, overlapping chunks to preserve context across boundaries.
- Writes each chunk, along with its associated metadata (source filename, title, chunk index, token positions), 
  to a JSONL file — one chunk per line.

Key Features:
- Automatically creates output directories if they do not exist.
- Uses overlapping windows to reduce information loss between adjacent chunks.
- Designed for scalable pre-processing of large text corpora for retrieval-augmented generation (RAG) pipelines.

Intended Usage:
Run this script directly to chunk all documents in the input directory and output them to a single JSONL file. 
These chunks are later embedded and indexed for similarity search in the FORAGER system.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import json
__docformat__ = "google"
import streamlit as st

# Contains all of the JSON docs


# Load specific tokenizer
@st.cache_resource
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct") # This is the same tokenizer that Groq uses

# Define the chunking function
def chunk_text(text, window_size=300, overlap=50):
    """
    This function chunks the text by the given window size and overlaps each chunk by the given overlap value.

    Args:
        text (str): The text to be chunked. It comes from the document's JSON, in the "content" field.
        window_size (int): The number of tokens per chunk. Default is 300.
        overlap (int): The number of tokens each chunk shares with its predecessor to avoid losing context between chunks. Default is 50.
    
    Returns:
        chunks (list): A list of dicts. Each dict contains the chunk text and metadata.
    """
    # Tokenize the content
    # Creates a list of all token IDs representing the entire doc's text
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Initialize an empty list to store the chunks
    chunks = []
    # Initialize variables to hold starting position and chunk position
    start = 0
    chunk_id = 0

    # Loop over content's tokens and chunk
    while start < len(tokens):
        # Identify the end of the chunk
        end = start + window_size
        # Grab one 300-token slice (or whatever window_size is) as a chunk
        chunk_tokens = tokens[start:end]
        # Decode tokens into text
        chunk_text = tokenizer.decode(chunk_tokens)

        # Package each chunk of text and metadata into a dict
        # and add it to the chunks list
        chunks.append({
            "chunk_id": chunk_id, # Unique index for this chunk
            "start_token": start, # Token position where this chunk begins in the full doc
            "end_token": min(end, len(tokens)), # Token position where this chunk ends (cannot go past total tokens)
            "text": chunk_text # Human-readable text for this chunk
        })

        chunk_id += 1
        # Move window forward with overlap
        start += window_size - overlap

    return chunks

def main():
    """
    This method loads each JSON doc from the corpus,
    pulls the "content" field, chunks the text, and
    saves the chunks into a .jsonl file (with one chunk per line).
    """
    # Directory from which the JSONs are coming
    base_dir = Path(__file__).resolve().parent.parent / "FORAGER_corpus" / "heterogenous_integration" / "json"
    all_docs = list(base_dir.glob("*.json"))
    # Create the output directory to store doc chunks
    output_path = Path("FORAGER_corpus/heterogenous_integration/chunks/chunks.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the chunks to the .json doc
    with open(output_path, "w", encoding="utf-8") as f_out:
        # Iterate through all the documents
        for doc_path in all_docs:
            # Load individual .json doc
            with open(doc_path, "r", encoding="utf-8") as f_in:
                doc = json.load(f_in)
            
            # Chunk the docs 
            chunks = chunk_text(doc["content"])

            for chunk in chunks:
                out = {
                    "source_filename": doc["source_filename"],
                    "title": doc.get("title", ""),
                    "chunk_id": chunk["chunk_id"],
                    "start_token": chunk["start_token"],
                    "end_token": chunk["end_token"],
                    "text": chunk["text"]
                }
                f_out.write(json.dumps(out) + "\n")

    print(f"[OK] Finished chunking {len(all_docs)} documents.")

# main()