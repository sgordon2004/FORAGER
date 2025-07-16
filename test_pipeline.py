from FORAGER.llm_extractor import extract_atomic_claims_llm
from FORAGER.bs import detect_bs
from FORAGER.embedder import search_database
from FORAGER.runner import get_llm_response

def full_forager_pipeline(question: str, k: int = 3):
    """
    Runs the full FORAGER pipeline:
    1. Retrieve documents.
    2. Generate LLM answer via RAG.
    3. Extract atomic claims.
    4. Evaluate claims using BS detector and confidence checker.
    5. Prompt Lock Loop decides to accept or regenerate.
    """
    # Step 1: Retrieve documents
    retrieved_docs = search_database(question, k)
    
    retrieved_docs_combined = "\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # Step 2: Feed question and documents to LLM
    prompt = f"""
    You are answering the following question using ONLY the provided context. DO NOT use any outside knowledge at all.

    Question: {question}

    Context:
    {retrieved_docs_combined}

    Answer:
    """

    # Step 3: Get the LLM's answer and run it through the BS detector and confidence checker
    answer = get_llm_response(prompt)
    bs_decision = detect_bs(answer, retrieved_docs)
    print(f"LLM answer: {answer} | Decision: {bs_decision}")



full_forager_pipeline("What are 5 benefits of using 3D HI?")
# question = "What are some benefits of using 3D HI?"
# print(search_database(question, 3))