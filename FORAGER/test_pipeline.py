import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FORAGER.llm_extractor import extract_atomic_claims_llm
from FORAGER.bs import detect_bs
from FORAGER.embedder import FAISSEmbedder
from FORAGER.runner import get_llm_response
from FORAGER.confidence import check_confidence

def generate_and_evaluate_claims(embedder: FAISSEmbedder, question: str, k: int = 3):
    """
    Retrieves context, generates an LLM answer, extracts atomic claims, and performs a single-pass evaluation.

    This function is the entry point for the initial stage of the FORAGER pipeline. It:
    1. Retrieves the top-k most relevant chunks from the FAISS knowledge base based on the input question.
    2. Passes the retrieved context and question to an LLM to generate an answer.
    3. Extracts atomic claims from the LLM’s answer using a claim extraction model.
    4. Evaluates each claim using:
        - A BS detector to classify factual accuracy (Supported, Unsupported, Contradicted).
        - A confidence checker to assign a confidence level (High, Medium, Low, or Zero).
    5. Returns the original LLM answer and a structured evaluation of the extracted claims.

    Note: This function does not run the full PLL (Prompt Locked Loop) pipeline. It only performs the initial pass.

    Args:
        embedder (FAISSEmbedder): An initialized embedder with access to the FAISS index.
        question (str): The user’s natural language query.
        k (int, optional): Number of top document chunks to retrieve. Default is 3.

    Returns:
        Tuple[str, dict]: 
            - `answer`: The LLM-generated response to the question.
            - `eval`: A dictionary where each key is an atomic claim string, and each value contains:
                - "label": BS detection result.
                - "confidence": Confidence rating.
                - "supporting_chunks": List of top-k chunks used for evaluation with accompanying metadata.
    """
    print("✅ Starting generate_and_evaluate_claims()...")
    embedder.initialize_faiss()
    print("✅ FAISS initialized in pipeline scope.")

    # Step 1: Retrieve documents
    try:
        retrieved_docs = embedder.search_database(question, k)
        print(f"✅ Retrieved {len(retrieved_docs)} docs")
    except Exception as e:
        print(f"❌ Error during search_database(): {type(e).__name__}: {e}")
        raise e
    
    retrieved_docs_text = [doc["text"] for doc in retrieved_docs]
    # print(f"Chunks used for answer: {retrieved_docs_text}")
    retrieved_docs_combined = "\n\n".join([doc["text"] for doc in retrieved_docs])

    # Note: I took this part out 
    # - If the question is **unanswerable from the context**, respond with: 
    # "This question cannot be answered by the information in the knowledge base."
    
    # Step 2: Feed question and documents to LLM
    prompt = f"""
    You are answering the following question using ONLY the provided context. 

    Guidelines:
    - Use the terminology and phrasing directly from the context.
    - Do NOT generalize to vague or abstract terms (e.g., do NOT replace "components" or "technologies" with "functions").
    - Do NOT mention the existence of a context or use phrases like "according to the context" or "according to the documents".
    - Do NOT mention source IDs, authors, document numbers, or citations.
    - Tailor your answer to the type of question being asked:
        - If the question asks for a **definition**, give a full and accurate definition using exact terms from the context.
        - If the question asks for an **explanation**, provide a clear explanation reflecting the context phrasing.
        - If the question asks for **examples, comparisons, or lists**, answer naturally while strictly staying within the context.


    Question: {question}

    Context:
    {retrieved_docs_combined}

    Answer:
    """

    # Step 3: Get the LLM's answer
    answer = get_llm_response(prompt)
    # Step 4: Extract atomic claims
    claims = extract_atomic_claims_llm(answer)

    # Step 5: Run all claims through BS detector
    eval = {}
    for claim in claims:
        label = detect_bs(embedder, claim, supporting_docs=retrieved_docs_text)
        eval[claim] = label

    # Run confidence checker
    updated_eval = {}
    for claim, label in eval.items():
        confidence = check_confidence(embedder, claim, label, retrieved_docs_text)
        updated_eval[claim] = {"label": label, "confidence": confidence, "supporting_chunks": retrieved_docs}
    
    # Update eval to store the confidence checker results
    eval = updated_eval
    print("💋")
    print(eval)

    return answer, eval