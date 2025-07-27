"""
This module handles embedding of textual chunks and managing a FAISS vector database for efficient similarity search. 
It is a core component of the FORAGER pipeline, enabling retrieval-augmented generation (RAG) by linking user queries 
to relevant chunks of information from a preprocessed corpus.

Core Functionality:
1. Embeds each chunk from a JSONL file (where each line is a JSON object containing a chunk of text) using the BGE embedding model.
2. Stores these embedded vectors in a FAISS database to enable fast nearest-neighbor search.
3. Provides a search function to retrieve the most relevant chunks for any given query.

Key Features:
- Supports initial creation of the FAISS database or loading from an existing index.
- Automatically updates the FAISS index if new chunks are appended to the source JSONL file.
- Includes utility functions for debugging and inspecting vectors.

Potential Extensions:
- Can be modified to embed chunks from alternate sources beyond the default 'chunks.jsonl'.
- Flexible enough to support future improvements in embedding models or chunking strategies.
"""
import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
__docformat__ = "google"

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from typing import List

# Important filepaths listed here
# This should set base_dir to be the outer FORAGER folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chunk_filepath = os.path.join(base_dir, "..", "FORAGER_corpus", "heterogenous_integration", "chunks", "chunks.jsonl")
faiss_db_filepath = os.path.join(base_dir, "vector_database", "index_db.faiss")

class FAISSEmbedder:
    """
    This class contains all of the variables and functions for the embedder.
    """
    def __init__(self, model_name = "BAAI/bge-base-en-v1.5", dim = 768, 
                 chunk_path = None, 
                 index_path = None):
        """
        Constructor for the embedder object.

        Arguments:
            model_name (str): Embedding model to use
            dim (int): Number of dimensions that embedded vectors have
            chunk_filepath (str): File path to chunk file
            faiss_db_filepath (str): File path to FAISS database
        """
        self.model = SentenceTransformer(model_name)
        self.prefix = "Represent this sentence for retrieval: "
        self.dim = dim
        self.chunk_filepath = os.path.abspath(chunk_path) if chunk_path else None
        self.faiss_db_filepath = os.path.abspath(index_path) if index_path else None
        self.faiss_db = faiss.IndexFlatIP(dim)

    @classmethod
    def create_default(cls):
        """
        Creates the embedder object using default chunk and FAISS database paths.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chunk_path = os.path.join(base_dir, "FORAGER_corpus", "heterogenous_integration", "chunks", "chunks.jsonl")
        index_path = os.path.join(base_dir, "vector_database", "index_db.faiss")
        return cls(chunk_path=chunk_path, index_path=index_path)

    def initialize_faiss(self):
        """
        This function ensures that the chunk and FAISS database filepaths exist. It then reads in the FAISS database
        if it exists or creates one if it doesn't.
        """
        os.makedirs(os.path.dirname(self.chunk_filepath), exist_ok = True)
        os.makedirs(os.path.dirname(self.faiss_db_filepath), exist_ok = True)

        if os.path.exists(self.faiss_db_filepath):
            self.faiss_db = faiss.read_index(self.faiss_db_filepath)
            print(f"\033[1;96mFAISS database exists. Size: {self.faiss_db.ntotal}\033[0m\n")
        else:
            print("Initial embed and build starting")
            self.initial_embed_and_build()
    
    def load_chunks(self):
        """
        This function loads in chunks from the specified chunk file.

        Returns: List of chunks contained in the chunk file
        """
        with open(self.chunk_filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def embed_chunks(self, chunks: List[dict]):
        """
        Embeds chunks from a specified list of chunks to embed

        Args: 
            chunks (list): List of chunks to embed
        Returns:
            embeddings (ndarray): A 2-D numpy array of vectors with each vector representing a chunk
        """

        texts = [self.prefix + chunk if isinstance(chunk, str) else chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True).astype("float32")
        print(f"\033[1;92m✅ {len(chunks)} chunks successfully embedded!\033[0m\n")
        return embeddings
    
    def embed_chunks_from_json(self):
        """
        Embeds all chunks currently in chunks.jsonl

        Args: 
            chunks (list): List of chunks to embed
        Returns:
            embeddings (ndarray): A 2-D numpy array of vectors with each vector representing a chunk
        """

        chunks = self.load_chunks()
        texts = [self.prefix + chunk if isinstance(chunk, str) else chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True).astype("float32")
        print(f"\033[1;92m✅ {len(chunks)} chunks successfully embedded!\033[0m\n")
        return embeddings

    def initial_embed_and_build(self):
        """
        This function sets up the FAISS database for the first time.
        """
        embeddings = self.embed_chunks_from_json()
        self.faiss_db.add(embeddings)
        faiss.write_index(self.faiss_db, self.faiss_db_filepath)
        print(f"\033[1;92m✅ FAISS database created and stored!\033[0m\n")

    def embed_text(self, text):
        """
        Embeds a single query or sentence using the BGE model.

        Arguments:
            text (str): Text to embed
        Returns:
            embedding (vector): Vector representing the text
        """
        embedding = self.model.encode([self.prefix + text], normalize_embeddings=True).astype("float32")[0]
        return embedding

    def search_database(self, query, top_k=3):
        """
        Function to search the FAISS database for the closest k chunks to the query (based on dot product).
        
        Args:
            query: User query or LLM response
            top_k: Number of closest vectors to return

        Returns:
            results (list): A list of dictionaries. Each dictionary gives the rank, source, order, score, and 
            text of each retrieved chunk.
        """
        chunks = self.load_chunks()
        query_vec = self.embed_text(query).reshape(1, -1)
        top_k = min(top_k, self.faiss_db.ntotal)
        scores, indices = self.faiss_db.search(query_vec, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            chunk = chunks[idx]
            result = {
                "rank": rank + 1,
                "source_filename": chunk["source_filename"],
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[0][rank]),
                "text": chunk["text"]
            }
            results.append(result)

        return results

    def add_to_faiss(self, new_embeddings):
        """
        Function to embed and add chunks from newly uploaded documents into FAISS

        Args:
            new_embedded_chunks: New vectors to add to FAISS database
        """
        self.faiss_db.add(new_embeddings)
        faiss.write_index(self.faiss_db, faiss_db_filepath)
        print(f"\033[1;96mFAISS database updated. New size: {self.faiss_db.ntotal}\033[0m\n")

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def get_faiss_db(self):
        """
        Returns the FAISS database for use in other files.
        """
        return self.faiss_db

    # Functions for debugging
    def clear_database(self):
        """
        Resets the FAISS database, if needed.
        """
        self.faiss_db.reset()
        print("\033[1;91mFAISS database cleared.\033[0m\n")

    def print_vector(self, index):
        """
        Prints vector at a certain location in the FAISS database, if needed.
        """
        vec = self.faiss_db.reconstruct(index)
        print(f"Vector at index {index}:\n{vec}\n")


