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
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
__docformat__ = "google"
import uuid
from dotenv import load_dotenv

from FORAGER.embedder import faiss_db, chunks, model, search_database # Importing loaded FAISS and chunks 
from FORAGER.pll_controller import run_pll_on_prompt

# clear existing locked_answers.json at start 
open("locked_answers.json", "w").close()


def main():
    print("Welcome to FORAGER Prompt-Lock Loop System!")

    while True: 
        user_prompt = input("\n Please enter your question (or type 'exit' to quit): ").strip()
        if user_prompt.lower() == "exit":
            print("Exiting FORAGER. Goodbye!")
            break
        
        print("\n Step 1: Getting relevant context...")
        context_chunks = search_database(user_prompt)
        print(f"Retrieved {len(context_chunks)} chunks.")
        for i, chunk in enumerate(context_chunks):
            print(f"Chunk {i+1}: {chunk[:100]}...") # Truncating for clarity
        
        print("\n Step 2: Running Prompt-Lock Loop...")
        # Combine context into one prompt string 
        full_prompt = f"{user_prompt}\n\nContext:\n" + "\n---\n".join(context_chunks)

        # Generate a unique ID for this question 
        question_id = str(uuid.uuid4())

        # Call PLL
        result = run_pll_on_prompt(full_prompt, question_id=question_id)

        print("\n Step 3: Final Result:")
        if result["status"] == "locked":
            print(f"Answer locked in: {result['candidate']}")
        elif result["status"] == "success":
            print(f"High-confidence answer: {result['candidate']}")
        else: 
            print(f"Could not lock answer. Best guess: \n{result['best_guess']['candidate']}")
        
        print("\n Full reasoning history:")
        for i, step in enumerate(result["history"], 1):
            print(f"\nAttempt {i}: ")
            print(f"- Candidate: {step['candidate']}")
            print(f"- Score: {step['score']}")
            print(f"- Eval Label: {step['eval_label']}")
            print(f"- Confidence: {step['confidence_label']}")
            print(f"- Label: {step['label']}")

if __name__ == "__main__":
    main()


# def retrieve_relevant_chunks(question, k=3):
#     # Embed query
#     prefix = "Represent this sentence for retrieval: "
#     query_with_prefix = [prefix + question]
#     query_emb = model.encode(query_with_prefix, normalize_embeddings = True).astype("float32")

#     # Limit k to number of indexed chunks 
#     k = min(k, faiss_db.ntotal)

#     # Search FAISS
#     scores, indices = faiss_db.search(query_emb, k)

#     # Extract the chunk texts for these indices 
#     relevant_chunks = [chunks[idx]["text"] for idx in indices[0]]
#     return relevant_chunks

# def build_prompt(context_chunks, question):
#     context_text = "\n".join(context_chunks)
#     prompt = f"Use the following information to answer the question:\n {context_text}\n\nQuestion: {question}\nAnswer:"
#     return prompt


# if __name__ == "__main__":
#     load_dotenv()
#     API_KEY = os.getenv("GROQ_API_KEY")
#     print(f"\n Using API key: {API_KEY[:8]}...\n")

#     task_prompt = input("Enter your prompt: ")

#     #Step 1 - Retrieve relevant context chunks from KB via vector search
#     print("\n Retrieving relevant chunks...")
#     context_chunks = retrieve_relevant_chunks(task_prompt)

#     #Step 2 - Build prompt for LLM 
#     print("\n Building prompt...")
#     prompt = build_prompt(context_chunks, task_prompt)

#     #Step 3 - Run PLL loop on this prompt (generate + self-check + improve)
#     print("\n Running Prompt Lock Loop...")
#     result = run_pll_on_prompt(prompt)

#     print("\n --- Final Answer ---")
#     print (result["candidate"] if result["status"] == "success" else result ["best_guess"]["candidate"])   
    

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
