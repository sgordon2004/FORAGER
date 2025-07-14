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
from FORAGER.runner import initial_run
from FORAGER.evaluator import run_eval_process
from FORAGER.loop_controller import prompt_lock_loop



# Entry point
if __name__ == "__main__":

    # Import your API key from .env file
    load_dotenv()
    API_KEY = os.getenv("GROQ_API_KEY")
    print(f"\n\033[1;96m🚀 Using API key: {API_KEY[:8]}...\033[0m\n")

    test_file = input("Enter the path/name of the test file (e.g., 4_distractors.json): ")
    print("\n")
    # Call initial_run() to feed first set of questions (Round 0)
    print("\033[1;94m=== 🧪 Initial Run: Sending questions to LLM ===\033[0m")
    initial_run(test_file)
    print("\n")

    # Call prompt_lock_loop() to initiate feedback loop
    print("\n\033[1;95m=== 🔁 Starting Prompt Lock Loop ===\033[0m")
    prompt_lock_loop(test_file)

    # Evaluate improvement