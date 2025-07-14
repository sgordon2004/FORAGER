"""
main.py

Entry point for the FORAGER system.

This script initiates the complete workflow:
- Loads and formats question batches
- Sends them to Groq's LLM
- Stores the LLM’s responses
- Evaluates the accuracy of responses
- Saves incorrectly answered questions for review

To run the full pipeline, execute this file directly:
    python main.py
"""

from dotenv import load_dotenv
import os
from FORAGER.embedder import faiss_db, chunks, model # Importing loaded FAISS and chunks 
# from FORAGER.runner import initial_run
# from FORAGER.evaluator import run_eval_process
# from FORAGER.loop_controller import prompt_lock_loop
from FORAGER.pll_controller import run_pll_on_prompt


# # Entry point
# if __name__ == "__main__":

#     # Import your API key from .env file
#     load_dotenv()
#     API_KEY = os.getenv("GROQ_API_KEY")
#     print(f"\n\033[1;96m🚀 Using API key: {API_KEY[:8]}...\033[0m\n")

#     test_file = input("Enter the path/name of the test file (e.g., 4_distractors.json): ")
#     print("\n")
#     # Call initial_run() to feed first set of questions (Round 0)
#     print("\033[1;94m=== 🧪 Initial Run: Sending questions to LLM ===\033[0m")
#     initial_run(test_file)
#     print("\n")

#     # Call prompt_lock_loop() to initiate feedback loop
#     print("\n\033[1;95m=== 🔁 Starting Prompt Lock Loop ===\033[0m")
#     prompt_lock_loop(test_file)

#     # Evaluate improvement

def retrieve_relevant_chunks(question, k=3):
    # Embed query
    prefix = "Represent this sentence for retrieval: "
    query_with_prefix = [prefix + question]
    query_emb = model.encode(query_with_prefix, normalize_embeddings = True).astype("float32")

    # Limit k to number of indexed chunks 
    k = min(k, faiss_db.ntotal)

    # Search FAISS
    scores, indices = faiss_db.search(query_emb, k)

    # Extract the chunk texts for these indices 
    relevant_chunks = [chunks[idx]["text"] for idx in indices[0]]
    return relevant_chunks

def build_prompt(context_chunks, question):
    context_text = "\n".join(context_chunks)
    prompt = f"Use the following information to answer the question:\n {context_text}\n\nQuestion: {question}\nAnswer:"
    return prompt


if __name__ == "main":
    load_dotenv()
    API_KEY = os.getenv("GROQ_API_KEY")
    print(f"\n Using API key: {API_KEY[:8]}...\n")

    question = input("Enter your prompt: ")

    #Step 1 - Retrieve relevant context chunks from KB via vector search
    context_chunks = retrieve_relevant_chunks(question)

    #Step 2 - Build prompt for LLM 
    prompt = build_prompt(context_chunks, question)

    #Step 3 - Run PLL loop on this prompt (generate + self-check + improve)
    result = run_pll_on_prompt(prompt)

    print("\n --- Final Answer ---")
    print (result["candidate"])   