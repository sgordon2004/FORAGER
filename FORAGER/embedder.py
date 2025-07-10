import os # Allows for interaction with underlying operating system, like accessing files
import fitz # PyMuPDF - can extract info from PDFs
from bs4 import BeautifulSoup # Scrapes info from webpages
from sentence_transformers import SentenceTransformer # Does the embedding
import numpy as np # For arrays
import faiss # Vector database

# 1. Prepare embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5") # Embedding model being used

def embed(texts):
    # The BGE model expects this prefix
    prefix = "Represent this sentence for retrieval: "
    # Adds the search criteria/user question. Sets normalizes vectors so that dot product is the same as cosine similarity
    return model.encode([prefix + t for t in texts], normalize_embeddings = True)

# # 2. Extract text from PDFs and HTMLs
# def extract_text_from_pdf(file_path):
#     doc = fitz.open(file_path) # Could use enumerate(doc) to get page numbers - not totally sure how to do this
#     # Combine text into one long string with newline characters separating each page
#     return "\n".join(page.get_text() for page in doc) 

# def extract_text_from_html(file_path):
#     with open(file_path, "r", encoding = "utf-8") as f:
#         # Parse the HTML document with BeautifulSoup's built-in parser
#         soup = BeautifulSoup(f.read(), "html.parser") 
#         # Gets all the visible text with a new line between text from different tags like <h1> and <p>
#         return soup.get_text(separator = "\n", strip = True)
    
# # 3. Chunk the text
# # Chunks are broken based on characters. We may want to change this to breaking based on tokens or sentences.
# # Syrr also has a chunking function, so we could use that one instead
# def chunk_text(text, max_chars = 500): # Chunk limit set to 500 characters
#     # Creates a list of non-empty lines without whitespace
#     lines = [line.strip() for line in text.split("\n") if line.strip()]
#     chunks = [] # List of final chunks to return
#     current = "" # Current chunk being built
#     for line in lines: 
#         if len(current) + len(line) < max_chars: # If chunk it not at character limit, keep building it
#             current += " " + line
#         else: # If chunk is at the limit, add it to the list of final chunks and start a new one with the current line
#             chunks.append(current.strip())
#             current = line
#     if current:
#         chunks.append(current.strip()) # At the end, add any leftover text to the list of chunks
#     return chunks

# # 4. Load and chunk all files
# # Takes a list of file paths (PDF or HTML) to extract and chunk the text from
# def load_and_chunk_files(file_paths):
#     all_chunks = [] # List of text chunks
#     metadata = [] # List of dictionaries with source filename and chunk content

#     for path in file_paths:
#         ext = os.path.splitext(path)[1].lower() # Gets the file extension
#         if ext == ".pdf":
#             text = extract_text_from_pdf(path)
#         elif ext in [".html", ".htm"]:
#             text = extract_text_from_html(path)
#         else:
#             continue # Skip if file isn't supported
        
#         chunks = chunk_text(text) # Calls chunking function
#         for chunk in chunks:
#             all_chunks.append(chunk)
#             # "source" is the filename without the extension
#             # "text" is the actual text in the chunk
#             metadata.append({"source": os.path.basename(path), "text": chunk})
    
#     return all_chunks, metadata

# 5. Embed chunks with FAISS and make an index
def build_index(chunks):
    embeddings = embed(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape(1)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings

# 6. Search Function
def search_index(query, index, metadata, k = 5):
    query_emb = embed([query])
    query_emb = np.array(query_emb).astype("float32")
    scores, indices = index.search(query_emb, k)

    print("f\n Results for: \"{query}\"")
    for rank, idx in enumerate(indices[0]):
        print(f"{rank+1}. [Source: {metadata[idx]['source']}] Score: {scores[0][rank]:.4f}")
        print(f"    {metadata[idx]['text']}\n")

# 7. Example Usage
if __name__ == "__main__":
    # Add paths to PDF and HTML files 
    files = ["example.pdf", "example_page.html"]
    
    chunks, metadata = load_and_chunk_files(files)
    index, _ = build_index(chunks)
    
    # Perform a search
    search_index("What is semantic search?", index, metadata)
