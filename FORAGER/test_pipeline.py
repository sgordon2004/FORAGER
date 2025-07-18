import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FORAGER.llm_extractor import extract_atomic_claims_llm
from FORAGER.bs import detect_bs
from FORAGER.embedder import search_database, initialize_faiss
from FORAGER.runner import get_llm_response
from FORAGER.confidence import check_confidence
from FORAGER.pll_controller import prompt_locked_loop
print("✅ test_pipeline.py successfully imported")

# Uncomment if running directly
# initialize_faiss()

def full_forager_pipeline(question: str, k: int = 3):
    """
    Runs the full FORAGER pipeline:
    1. Retrieve documents.
    2. Generate LLM answer via RAG.
    3. Extract atomic claims.
    4. Evaluate claims using BS detector and confidence checker.
    5. Prompt Lock Loop decides to accept or regenerate.
    """
    print("✅ Starting full_forager_pipeline()...")
    initialize_faiss()
    print("✅ FAISS initialized in pipeline scope.")

    # Step 1: Retrieve documents
    try:
        retrieved_docs = search_database(question, k)
        print(f"✅ Retrieved {len(retrieved_docs)} docs")
    except Exception as e:
        print(f"❌ Error during search_database(): {type(e).__name__}: {e}")
        raise e
    
    retrieved_docs_text = [doc["text"] for doc in retrieved_docs]
    print(f"Chunks used for answer: {retrieved_docs_text}")
    retrieved_docs_combined = "\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # Step 2: Feed question and documents to LLM
    prompt = f"""
    You are answering the following question using ONLY the provided context. DO NOT use any outside knowledge at all.

    Question: {question}

    Context:
    {retrieved_docs_combined}

    Answer:
    """

    # Step 3: Get the LLM's answer
    print(f"Prompt to LLM:\n{prompt}")
    answer = get_llm_response(prompt)
    print(f"✅ LLM Answer:\n{answer}")

    # Step 4: Extract claims from LLM answer
    claims = extract_atomic_claims_llm(answer) # a list of all the atomic claims made by the LLM

    # Step 5: Run all claims through BS detector
    eval = {} # dict to map claim to eval_label
    for claim in claims:
        label = detect_bs(claim, supporting_docs=retrieved_docs_text)
        eval[claim] = label

    # Run confidence checker
    updated_eval = {}
    for claim, label in eval.items():
        confidence = check_confidence(claim, label, retrieved_docs)
        updated_eval[claim] = {"label": label, "confidence": confidence}
    
    eval = updated_eval
    print(eval)

    return answer, eval, retrieved_docs_text


dummy_eval = [
    {"claim": "3D HI reduces wiring length.", "label": "Unsupported", "confidence": "Medium"},
    {"claim": "3D HI increases flexibility.", "label": "Supported", "confidence": "Low"},
]
from ingestor import extract_all_pdfs
from chunker import main
dummy_docs = ["3D HI allows better interconnect density.", "It may lead to better performance due to shorter connections."]
extract_all_pdfs()
main()
initialize_faiss()

prompt_locked_loop("What are some benefits of 3D HI?", dummy_eval, dummy_docs)