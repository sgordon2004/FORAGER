"""
This module loads each document's JSON file, tokenizes the content, and chunks it.
It then wraps each chunk with metada and saves it as a JSONL.
"""

from pathlib import Path
import json
from transformers import AutoTokenizer
__docformat__ = "google"
# Directory from which the JSONs are coming
input_dir = Path("FORAGER_corpus/heterogenous_integration/json")
# Contains all of the JSON docs
all_docs = list(input_dir.glob("*.json"))

# Load specific tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct") # This is the same tokenizer that Groq uses

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
    This method loads each JOSN doc from the corpus,
    pulls the "content" field, chunks the text, and
    saves the chunks into a .jsonl file (with one chunk per line).
    """
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

main()